---
title: GitOps for ML model deployments with Argo CD
date: 2026-06-15
summary: How I deploy model-serving containers with Argo CD, promote model versions by pull request, roll back with git revert and keep secrets out of Git
tags: MLOps, DevSecOps
---

Shipping a new model version is a deployment like any other, but it is easy to end up with a process where someone runs `kubectl apply` from a laptop or a training job pushes straight to production. Then nobody can say which model is live, who approved it, or how to get back to the previous one.

GitOps fixes that by making a Git repository the single description of what should be running. This post is the layout and set of manifests I use to deploy model-serving containers with [Argo CD](https://argo-cd.readthedocs.io/).

## What GitOps means here

The idea is simple:

1. The desired state of the cluster (Deployments, Services, config) lives in Git as declarative manifests.
2. A controller running in the cluster, Argo CD in this case, continuously compares the live state with Git and applies the difference.
3. Every change, including "deploy model version 8", is a commit. Review happens in pull requests, and the Git history is the audit log.

The important shift is that CI no longer pushes to the cluster. CI builds and tests artefacts, then proposes a change to Git. The cluster pulls.

## Repo layout: app code versus config

The Argo CD docs recommend keeping manifests in a separate repository from application source code. The reasons that matter most for ML work: you can change config without triggering a full build, the config history stays readable, and you can give people commit access to code without giving them the ability to change production. It also stops CI from triggering itself in a loop when it commits manifest changes.

I use two repositories:

```text
model-service/                 # app repo: code, Dockerfile, tests, CI
  src/serve.py
  Dockerfile
  .github/workflows/build.yaml

ml-deploy-config/              # config repo: what runs where
  apps/
    churn-model/
      base/
        deployment.yaml
        service.yaml
        mlflow-credentials.yaml
        kustomization.yaml
      overlays/
        staging/
          kustomization.yaml
        production/
          kustomization.yaml
  argocd/
    churn-model-staging.yaml
    churn-model-production.yaml
```

The base holds everything common. Each overlay pins the image and model version for that environment. I use Kustomize because Argo CD supports it natively and it keeps overlays small; Helm works just as well.

## The model-serving Deployment

This is the base Deployment. The container is a model server that loads a model from the registry at start-up, identified by an environment variable.

```yaml
# apps/churn-model/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-model
  labels:
    app: churn-model
spec:
  replicas: 2
  selector:
    matchLabels:
      app: churn-model
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  template:
    metadata:
      labels:
        app: churn-model
    spec:
      containers:
        - name: server
          image: ghcr.io/example/churn-model-server:placeholder
          ports:
            - containerPort: 8080
          env:
            - name: MODEL_URI
              value: "models:/churn-classifier/7"
            - name: MLFLOW_TRACKING_URI
              value: "https://mlflow.internal.example.com"
            - name: MLFLOW_TRACKING_TOKEN
              valueFrom:
                secretKeyRef:
                  name: mlflow-credentials
                  key: token
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            periodSeconds: 10
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              memory: "2Gi"
```

Two choices here are deliberate:

- **The readiness probe must reflect model loading.** If `/ready` returns 200 before the model is in memory, a rolling update will send traffic to pods that cannot answer. With `maxUnavailable: 0`, old pods stay up until new ones are ready, so a model that fails to load never takes traffic.
- **The model URI is a pinned version, not an alias.** More on that in the registry section.

The base `kustomization.yaml` just lists the resources:

```yaml
# apps/churn-model/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
  - mlflow-credentials.yaml
```

The `mlflow-credentials.yaml` file is an `ExternalSecret`, covered in the secrets section below.

And the production overlay pins the image and the model version:

```yaml
# apps/churn-model/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ml-production
resources:
  - ../../base
images:
  - name: ghcr.io/example/churn-model-server
    newTag: "1.4.2"
patches:
  - target:
      kind: Deployment
      name: churn-model
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/env/0/value
        value: "models:/churn-classifier/7"
```

## The Argo CD Application

An `Application` tells Argo CD which path in which repo to sync to which cluster and namespace. This is the production one:

```yaml
# argocd/churn-model-production.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: churn-model-production
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: ml-models
  source:
    repoURL: https://github.com/example/ml-deploy-config.git
    targetRevision: main
    path: apps/churn-model/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

What each part does:

- `automated` makes Argo CD sync whenever `main` changes. `prune: true` deletes resources that were removed from Git, and `selfHeal: true` reverts manual changes made directly in the cluster. Both default to `false`, and I turn both on so Git really is the source of truth.
- The finalizer means deleting the Application also deletes the resources it manages. Leave it out if you would rather they stay behind.
- `project: ml-models` refers to an Argo CD `AppProject`, which restricts which repos and namespaces this Application may touch. Use a dedicated project rather than `default` so a mistake in one team's config cannot deploy into another team's namespace.

The staging Application is the same with a different name, path and namespace. Once you have more than a handful of these, an `ApplicationSet` can generate them from a template.

## Promoting a new model version by pull request

With this layout, promotion is a small diff. Say version 8 of `churn-classifier` has passed offline evaluation and is registered. The flow I use:

1. CI in the app repo builds and tests the server image (if the code changed) and pushes it with an immutable tag.
2. A promotion job opens a pull request against `ml-deploy-config` that changes the staging overlay:

```bash
cd ml-deploy-config/apps/churn-model/overlays/staging
kustomize edit set image ghcr.io/example/churn-model-server:1.4.3
# and update the MODEL_URI patch to models:/churn-classifier/8
git checkout -b promote-churn-v8-staging
git commit -am "churn-model staging: model v8, server 1.4.3"
git push origin promote-churn-v8-staging
```

3. After merge, Argo CD syncs staging. Smoke tests and a shadow or canary comparison run against it.
4. A second pull request makes the same change to the production overlay. Branch protection on `main` requires approval from the people who own production, and the PR description links the evaluation report and the registry entry.

The PR is the change request. Reviewers see exactly which model and image are moving to which environment, and nothing reaches production without a merged commit.

## Rollback by git revert

Rolling back is a revert of the promotion commit:

```bash
git revert <promotion-commit-sha>
git push origin main
```

Argo CD sees the new commit and syncs the previous image and model URI. The rollback itself is reviewed, recorded and repeatable.

Argo CD also has `argocd app rollback`, which redeploys an earlier synced revision, but the docs note that rollback cannot be performed against an application with automated sync enabled. That is another reason I treat `git revert` as the standard rollback: it works with auto-sync and keeps Git and the cluster in agreement. If you need speed in an incident, have the revert PR template ready and allow a smaller set of approvers for reverts.

## Secrets handling

Never commit plain Kubernetes Secrets to the config repo. Base64 is encoding, not encryption.

Argo CD's own documentation strongly recommends populating secrets on the destination cluster rather than injecting them during manifest generation. Generation-based plugins mean Argo CD needs access to the secrets, and it stores generated manifests in plaintext in its Redis cache. The recommended options include Sealed Secrets, the External Secrets Operator and the Secrets Store CSI Driver.

I use the External Secrets Operator so Git contains only a reference to the secret, not the value:

```yaml
# apps/churn-model/base/mlflow-credentials.yaml
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: mlflow-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: mlflow-credentials
  data:
    - secretKey: token
      remoteRef:
        key: ml/churn-model/mlflow
        property: token
```

Argo CD syncs the `ExternalSecret`, the operator fetches the value from the secret store and creates the Kubernetes Secret, and the Deployment reads it through `secretKeyRef`. Rotating a credential no longer needs a deployment at all.

## How it fits with a model registry

The model registry and the config repo answer different questions. The registry, MLflow in this example, records **which model versions exist**, with their metrics, lineage and artefacts. The config repo records **which version runs where**.

That is why I pin `models:/churn-classifier/7` rather than an alias such as `models:/churn-classifier@champion`. MLflow aliases are mutable pointers. If the server resolves an alias at start-up, moving the alias in the registry changes what production serves the next time a pod restarts, with no commit, no review and no clean revert. Pinning the version keeps the deployment reproducible.

Aliases are still useful as a signal: a registry webhook or scheduled job can watch for `champion` moving and open the promotion PR automatically. The alias proposes a change and the pull request approves it.

For the same reason, pin images by tag at minimum, or by digest if your registry allows tags to be overwritten. Kustomize's `images` field accepts a `digest` instead of `newTag`.

## Sources

- [Argo CD documentation](https://argo-cd.readthedocs.io/en/stable/)
- [Argo CD best practices (separating config and source repos)](https://argo-cd.readthedocs.io/en/stable/user-guide/best_practices/)
- [Argo CD declarative setup and Application spec](https://argo-cd.readthedocs.io/en/stable/operator-manual/declarative-setup/)
- [Argo CD automated sync policy](https://argo-cd.readthedocs.io/en/stable/user-guide/auto_sync/)
- [Argo CD secret management](https://argo-cd.readthedocs.io/en/stable/operator-manual/secret-management/)
- [Argo CD Kustomize support](https://argo-cd.readthedocs.io/en/stable/user-guide/kustomize/)
- [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Kubernetes liveness, readiness and startup probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [External Secrets Operator](https://external-secrets.io/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
