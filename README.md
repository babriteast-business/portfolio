# Babrit Behera, portfolio

Static site, two pages: **Home** (`index.html`) and **Blog** (`blog.html`, posts open in `post.html`).
No build step. Three.js and GSAP are bundled in `assets/vendor`, so nothing loads from a CDN except the fonts.

## Add a blog post (PowerShell, from this folder)
```powershell
.\blog.ps1 new "My post title" "RAG, Agents"
```
Notepad opens the new file in `posts\`. Write it in Markdown, save, then:
```powershell
.\blog.ps1 build
git add .
git commit -m "New post"
git push
```
Render republishes the site in about a minute. To hide a post, set `draft: true` at the top of its file and run `.\blog.ps1 build`.
If PowerShell blocks the script, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Fill in your contact details
Search `index.html` for `XXXX` and replace: email (2 places + the Copy button), phone (2 places + Copy), GitHub URL (2 places).

## Preview on your PC
```powershell
python -m http.server 8080
```
Open http://localhost:8080. Opening `index.html` by double-click won't load the blog posts, because browsers block that for local files.
