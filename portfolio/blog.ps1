<#
  Blog helper for the portfolio. Run from the portfolio folder.

    .\blog.ps1 new "My post title" "RAG, Agents"   # creates posts\my-post-title.md and opens it
    .\blog.ps1 build                               # rebuilds posts\index.json from all posts

  Every post is a Markdown file in posts\ that starts with:
    ---
    title: My post title
    date: 2026-11-08
    summary: One sentence shown on the blog list.
    tags: RAG, Agents
    draft: false
    ---
  Set draft: true to hide a post until it's ready.
#>
param([string]$Command = "build", [string]$Title = "", [string]$Tags = "")
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$posts = Join-Path $root "posts"
$utf8  = New-Object System.Text.UTF8Encoding $false

function Get-Slug([string]$text) {
  $s = $text.ToLower() -replace "[^a-z0-9\s-]", "" -replace "\s+", "-" -replace "-+", "-"
  return $s.Trim("-")
}

function Build-Index {
  $items = @()
  Get-ChildItem -Path $posts -Filter *.md | Sort-Object Name | ForEach-Object {
    $raw = [System.IO.File]::ReadAllText($_.FullName, $utf8).TrimStart([char]0xFEFF)
    $meta = @{}
    $body = $raw
    $m = [regex]::Match($raw, "^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?")
    if ($m.Success) {
      foreach ($line in ($m.Groups[1].Value -split "\r?\n")) {
        $kv = [regex]::Match($line, "^(\w+):\s*(.*)$")
        if ($kv.Success) { $meta[$kv.Groups[1].Value] = $kv.Groups[2].Value.Trim() }
      }
      $body = $raw.Substring($m.Length)
    }
    $words = ($body -split "\s+" | Where-Object { $_ -ne "" }).Count
    $tagList = @()
    if ($meta["tags"]) { $tagList = @($meta["tags"] -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }) }
    $title = $meta["title"]; if (-not $title) { $title = $_.BaseName }
    $date  = $meta["date"];  if (-not $date)  { $date = $_.LastWriteTime.ToString("yyyy-MM-dd") }
    $items += [pscustomobject]@{
      slug    = $_.BaseName
      title   = $title
      date    = $date
      summary = [string]$meta["summary"]
      tags    = $tagList
      minutes = [math]::Max(1, [math]::Round($words / 220))
      draft   = ($meta["draft"] -eq "true")
    }
  }
  $json = ConvertTo-Json -InputObject @($items) -Depth 4
  [System.IO.File]::WriteAllText((Join-Path $posts "index.json"), $json, $utf8)
  $live = @($items | Where-Object { -not $_.draft }).Count
  Write-Host "posts\index.json updated: $($items.Count) posts ($live published)." -ForegroundColor Green
}

switch ($Command.ToLower()) {
  "new" {
    if (-not $Title) { $Title = Read-Host "Post title" }
    if (-not $Tags)  { $Tags  = Read-Host "Tags, comma separated (for example: RAG, Agents)" }
    $slug = Get-Slug $Title
    if (-not $slug) { throw "The title needs at least one letter or number." }
    $file = Join-Path $posts "$slug.md"
    if (Test-Path $file) { throw "posts\$slug.md already exists. Pick another title or edit that file." }
    $today = Get-Date -Format "yyyy-MM-dd"
    $text = @"
---
title: $Title
date: $today
summary: One sentence that says what this post is about.
tags: $Tags
draft: false
---

Start writing here. Markdown works: **bold**, lists, links, and code blocks.

## A section heading

Your text.
"@
    [System.IO.File]::WriteAllText($file, $text, $utf8)
    Write-Host "Created posts\$slug.md" -ForegroundColor Green
    Build-Index
    Start-Process notepad.exe $file
  }
  "build" { Build-Index }
  default { Write-Host 'Use: .\blog.ps1 new "Title" "Tag1, Tag2"   or   .\blog.ps1 build' }
}
