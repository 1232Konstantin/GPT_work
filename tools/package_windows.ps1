$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

# An installed Windows Tesseract is required on the BUILD machine only.
# End users receive a fully bundled copy.
$tesseractDir = Join-Path $env:ProgramFiles 'Tesseract-OCR'
$exe = Join-Path $tesseractDir 'tesseract.exe'
if (-not (Test-Path $exe)) { throw "Windows Tesseract not found at $exe. Install choco install tesseract -y first." }
$vendorDir = Join-Path $root 'vendor\tesseract'
New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
Copy-Item -Path (Join-Path $tesseractDir '*') -Destination $vendorDir -Recurse -Force
$tessdataDir = Join-Path $vendorDir 'tessdata'
New-Item -ItemType Directory -Force -Path $tessdataDir | Out-Null
foreach ($lang in @('jpn','jpn_vert')) {
    $model = Join-Path $tessdataDir "$lang.traineddata"
    if (-not (Test-Path $model)) {
        $url = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/$lang.traineddata"
        Invoke-WebRequest -Uri $url -OutFile $model
    }
    if ((Get-Item $model).Length -lt 100000) { throw "OCR model missing or incomplete: $model" }
}
& (Join-Path $vendorDir 'tesseract.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'Bundled Tesseract cannot execute.' }

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies installation failed' }
python -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed' }
# Start both real Qt windows headlessly on the Windows build host.
$env:QT_QPA_PLATFORM = 'offscreen'
python main.py --gui-self-test
if ($LASTEXITCODE -ne 0) { throw 'Qt GUI smoke test failed before packaging' }
python tools/create_icon.py
if ($LASTEXITCODE -ne 0) { throw 'Icon creation failed' }
python -m PyInstaller --noconfirm --clean --windowed --onedir `
    --name MangaOfflineTranslator --icon assets\manga.ico `
    --add-data 'vendor\tesseract;vendor\tesseract' main.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }
$bundle = Join-Path $root 'dist\MangaOfflineTranslator\MangaOfflineTranslator.exe'
& $bundle --self-test
if ($LASTEXITCODE -ne 0) { throw 'Bundled executable self-test failed' }
& $bundle --gui-self-test
if ($LASTEXITCODE -ne 0) { throw 'Bundled Qt GUI smoke test failed' }

$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw 'Inno Setup 6 missing; choco install innosetup -y' }
& $iscc installer\MangaOfflineTranslator.iss
if ($LASTEXITCODE -ne 0) { throw 'Installer compile failed' }
$setup = Get-ChildItem release\MangaOfflineTranslator_Setup_*_win64.exe | Select-Object -First 1
if (-not $setup) { throw 'Installer not created' }
Write-Host "INSTALLER: $($setup.FullName)"
Get-FileHash $setup.FullName -Algorithm SHA256