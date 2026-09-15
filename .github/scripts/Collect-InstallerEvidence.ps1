# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

<#
.SYNOPSIS
    Record what an Unsloth install left behind, so two commits can be compared.

.DESCRIPTION
    Runs after install.ps1 and writes three files that compare_installer_evidence.py consumes:

      shortcuts.json   every .lnk the installer created, read back through the shell, with the
                       fields that constitute the launch contract: target, arguments (which carry
                       -WindowStyle and -ExecutionPolicy), working directory, window style, icon.
      artifacts.json   the installed files, with the content of the two whose content is a contract
                       (launch-studio.ps1 and unsloth.cmd), plus which files a second install run
                       rewrote.
      (the transcript is captured by the workflow, which owns the installer's stdout)

    Reading the .lnk files back through the shell rather than trusting the installer's own log is
    the point: a change to the launch transport is invisible in the transcript, and the transcript
    is what a prose review looks at.

    This script never fails the job. A collection error is recorded in the JSON as an error field so
    the comparer can call the run VOID, which is a different and more honest outcome than a step
    that went red for a reason nobody reads.
#>

[CmdletBinding()]
param(
    # Where the installer put Studio.
    [Parameter(Mandatory = $true)][string]$StudioHome,

    # Where to write the evidence.
    [Parameter(Mandatory = $true)][string]$OutDir,

    # Comma-free list of directories to search for .lnk files. Defaults to the desktop and the
    # user's Start Menu programs folder, which is where install.ps1 writes them.
    [string[]]$ShortcutRoot,

    # Set when this is the second install of the same commit: the manifest from the first run is
    # read back and the two compared, so "a reinstall writes nothing" is measured rather than
    # assumed. Several changes in this area touch content-comparison paths.
    [string]$CompareAgainst
)

$ErrorActionPreference = 'Continue'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ---------------------------------------------------------------------------
# Shortcuts
# ---------------------------------------------------------------------------

if (-not $ShortcutRoot -or $ShortcutRoot.Count -eq 0) {
    $ShortcutRoot = @(
        [Environment]::GetFolderPath('Desktop'),
        [Environment]::GetFolderPath('CommonDesktopDirectory')
    )
    # Guarded rather than assumed: Join-Path throws on a null Path, and %APPDATA% is unset off
    # Windows, so composing it unconditionally turned a parse-clean script into one that printed a
    # binding error before doing any work. Observed running this under pwsh on Linux.
    if ($env:APPDATA) {
        $ShortcutRoot += (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
    }
}

$shortcuts = New-Object System.Collections.ArrayList
$shell = $null
try {
    # WScript.Shell is the only shortcut reader available to Windows PowerShell 5.1 without
    # IShellLink interop, and it is what the installer itself uses to write them.
    $shell = New-Object -ComObject WScript.Shell
} catch {
    [void]$shortcuts.Add([ordered]@{
        name  = '<collection failed>'
        error = "could not create WScript.Shell: $($_.Exception.Message)"
    })
}

if ($shell) {
    foreach ($root in ($ShortcutRoot | Where-Object { $_ })) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $found = @(Get-ChildItem -LiteralPath $root -Filter '*.lnk' -Recurse -ErrorAction SilentlyContinue |
                   Where-Object { $_.Name -like '*nsloth*' })
        foreach ($file in $found) {
            try {
                $lnk = $shell.CreateShortcut($file.FullName)
                # Keyed on the file name, not the full path: the two sides run in different
                # workspaces, so a path would differ for a reason that is not a behaviour change.
                [void]$shortcuts.Add([ordered]@{
                    name             = $file.Name
                    root             = (Split-Path $root -Leaf)
                    targetPath       = $lnk.TargetPath
                    arguments        = $lnk.Arguments
                    workingDirectory = $lnk.WorkingDirectory
                    windowStyle      = [string]$lnk.WindowStyle
                    iconLocation     = $lnk.IconLocation
                    description      = $lnk.Description
                })
            } catch {
                [void]$shortcuts.Add([ordered]@{
                    name  = $file.Name
                    error = "could not read: $($_.Exception.Message)"
                })
            }
        }
    }
}

# -Depth, because ConvertTo-Json truncates at 2 by default and silently emits the type name instead
# of the object. An evidence file that says "System.Collections.Hashtable" compares equal to another
# one that says the same thing, which is a green run that measured nothing.
#
# Always an array, even for one entry. ConvertTo-Json unwraps a single-element collection into a
# bare object, and -AsArray does not exist on Windows PowerShell 5.1, so the shape of this file
# would depend on how many shortcuts happened to exist. The consumer also tolerates an object, but
# evidence whose schema varies with its contents is a trap either way.
$shortcutJson = $shortcuts | ConvertTo-Json -Depth 6
if ($shortcuts.Count -le 1) { $shortcutJson = "[$($shortcutJson)]" }
Set-Content -LiteralPath (Join-Path $OutDir 'shortcuts.json') -Value $shortcutJson -Encoding utf8

Write-Host "collected $($shortcuts.Count) shortcut(s)"
foreach ($s in $shortcuts) {
    Write-Host ("  {0}: {1} {2}" -f $s.name, $s.targetPath, $s.arguments)
}

# ---------------------------------------------------------------------------
# Installed files
# ---------------------------------------------------------------------------

# Content is compared for these two only. They are generated by the installer from templates, so
# their text is a contract; everything else is either a downloaded binary (compares by hash, which
# drifts with upstream) or a venv (tens of thousands of files, and not what this lane is about).
$contentFiles = @('launch-studio.ps1', 'unsloth.cmd', 'share\studio.conf')

$files = [ordered]@{}
$artifactError = $null

if (Test-Path -LiteralPath $StudioHome) {
    foreach ($relative in $contentFiles) {
        $full = Join-Path $StudioHome $relative
        if (-not (Test-Path -LiteralPath $full)) { continue }
        try {
            $files[$relative] = [ordered]@{
                content = (Get-Content -Raw -LiteralPath $full -ErrorAction Stop)
                sha256  = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
            }
        } catch {
            $files[$relative] = [ordered]@{ error = $_.Exception.Message }
        }
    }
    # The shape of the tree, without its contents: a change that stops creating a directory, or
    # starts creating one, shows up here and nowhere else.
    try {
        $tops = @(Get-ChildItem -LiteralPath $StudioHome -ErrorAction Stop |
                  Select-Object -ExpandProperty Name | Sort-Object)
        foreach ($name in $tops) {
            if (-not $files.Contains($name)) { $files[$name] = [ordered]@{ present = $true } }
        }
    } catch {
        $artifactError = "could not enumerate ${StudioHome}: $($_.Exception.Message)"
    }
} else {
    $artifactError = "the install root $StudioHome does not exist, so the installer did not finish"
}

# ---------------------------------------------------------------------------
# Idempotency, when asked
# ---------------------------------------------------------------------------

$rewritten = $null
if ($CompareAgainst) {
    if (-not (Test-Path -LiteralPath $CompareAgainst)) {
        # Deliberately NOT an empty list. An empty list means "measured, nothing was rewritten",
        # and $null means "not measured"; the comparer treats those differently and must.
        Write-Host "::warning::no first-run manifest at $CompareAgainst, so idempotency was not measured"
    } else {
        $rewritten = New-Object System.Collections.ArrayList
        try {
            $before = Get-Content -Raw -LiteralPath $CompareAgainst | ConvertFrom-Json
            foreach ($relative in $contentFiles) {
                $b = $before.files.$relative
                $a = $files[$relative]
                if (-not $b -and -not $a) { continue }
                if (-not $b -or -not $a) { [void]$rewritten.Add($relative); continue }
                if ($b.sha256 -and $a.sha256 -and $b.sha256 -ne $a.sha256) {
                    [void]$rewritten.Add($relative)
                }
            }
        } catch {
            Write-Host "::warning::could not compare against $CompareAgainst : $($_.Exception.Message)"
            $rewritten = $null
        }
    }
}

$artifacts = [ordered]@{
    studioHome = $StudioHome
    files      = $files
}
if ($artifactError) { $artifacts['error'] = $artifactError }
if ($null -ne $rewritten) { $artifacts['rewrittenOnSecondRun'] = @($rewritten) }

$artifacts | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutDir 'artifacts.json') -Encoding utf8

Write-Host "collected $($files.Count) artifact entr(ies)$(if ($artifactError) { " [$artifactError]" })"
if ($null -ne $rewritten) {
    Write-Host "second run rewrote: $(if ($rewritten.Count) { $rewritten -join ', ' } else { 'nothing' })"
}

# Always zero. A collection failure is data for the comparer, not a red step.
exit 0
