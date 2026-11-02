#define MyAppName "Manga Offline Translator"
#define MyAppVersion "0.6.0"
#define MyAppPublisher "Manga Offline Translator Project"
#define MyAppExeName "MangaOfflineTranslator.exe"
[Setup]
AppId={{6D7AB45D-66D4-43BC-A547-87CFF817C963}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\MangaOfflineTranslator
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\release
OutputBaseFilename=MangaOfflineTranslator_Setup_{#MyAppVersion}_win64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\manga.ico
UninstallDisplayIcon={app}\MangaOfflineTranslator.exe
DisableProgramGroupPage=yes
ChangesAssociations=no
CloseApplications=yes
[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"
[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительно:"; Flags: unchecked
[Files]
Source: "..\dist\MangaOfflineTranslator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить {#MyAppName}"; Flags: nowait postinstall skipifsilent