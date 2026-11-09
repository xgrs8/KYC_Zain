; A.L.G - Automatic Label Generator - Inno Setup script (plan 10.2)
; Per-user install (no admin): {localappdata}\RFALG\ALG
; Build:  iscc ALG\packaging\alg.iss   (from the ALG folder, after PyInstaller)
; Silent: ALG-Setup-x64.exe /SILENT (or /VERYSILENT); uninstall silently:
;         unins000.exe /SILENT

#define MyAppName "A.L.G - Automatic Label Generator"
#define MyAppVersion "0.1.0"
#define MyAppExeName "ALG.exe"
#define MyAppIcon "packaging\icons\ALG.ico"

[Setup]
AppId={{5D1E8B29-7C4A-4F62-B9D3-86A0C2F47E15}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=RFALG
DefaultDirName={localappdata}\RFALG\ALG
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
SourceDir=..
OutputDir=dist
OutputBaseFilename=ALG-Setup-x64
SetupIconFile={#MyAppIcon}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
DisableWelcomePage=yes
DisableDirPage=yes
DisableReadyPage=yes
DisableFinishedPage=yes
LicenseFile=
InfoBeforeFile=

[Files]
; PyInstaller one-folder output (build with packaging\build.ps1 first)
Source: "dist\ALG\ALG.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\ALG\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove the app itself; the user's workspace
; (%USERPROFILE%\Documents\RFALG) and logs (%LOCALAPPDATA%\RFALG\logs)
; are intentionally KEPT (they contain the user's data).
Type: filesandordirs; Name: "{app}"
