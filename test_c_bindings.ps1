$moduleAlreadyLoaded = Get-Module Microsoft.VisualStudio.DevShell
if ($moduleAlreadyLoaded) {
    Write-Output "DevShell module is already loaded, skipping."
} else {
    $assemblyPresent = [System.AppDomain]::CurrentDomain.GetAssemblies() | Where-Object { $_.FullName -like '*Microsoft.VisualStudio.DevShell*' }
    if ($assemblyPresent) {
        Write-Output "DevShell assembly is already loaded, skipping."
    } else {
        if (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe") {
            Write-Error "Visual Studio is not installed, please install it first."
            return
        }
        $vsPath = &"${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationpath
        $commonLocation = "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
        if (Test-Path $commonLocation) {
            $dllPath = $commonLocation
        } else {
            $dllPath = (Get-ChildItem $vsPath -Recurse -File -Filter Microsoft.VisualStudio.DevShell.dll).FullName
        }
        Import-Module -Force $dllPath
        Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64"
    }
}

if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust is not installed, please install it first."
    return
}
if (!(Get-Command cbindgen -ErrorAction SilentlyContinue)) {
    Write-Error "cbindgen is not installed, please install it first."
    return
}

cargo build --release
cbindgen --config .\cbindgen.toml --crate onnxrust_proto --output bindings/c/onnxrust_proto.h --lang c
cl /nologo /W4 /O2 .\binding_test.c .\target\release\libonnxrust_proto.dll.lib
$env:PATH += ";$pwd\target\release"
.\binding_test.exe
Remove-Item .\binding_test.exe
Remove-Item .\binding_test.obj
