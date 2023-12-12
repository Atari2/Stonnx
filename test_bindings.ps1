$moduleAlreadyLoaded = Get-Module Microsoft.VisualStudio.DevShell
if ($moduleAlreadyLoaded) {
    Write-Output "DevShell module is already loaded, skipping."
}
else {
    $assemblyPresent = [System.AppDomain]::CurrentDomain.GetAssemblies() | Where-Object { $_.FullName -like '*Microsoft.VisualStudio.DevShell*' }
    if ($assemblyPresent) {
        Write-Output "DevShell assembly is already loaded, skipping."
    }
    else {
        if (!(Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe")) {
            Write-Error "Visual Studio is not installed, please install it first (make sure to install the C++ and C# workloads)."
            Write-Error "https://visualstudio.microsoft.com/downloads/"
            return
        }
        $vsPath = &"${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationpath
        $commonLocation = "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
        if (Test-Path $commonLocation) {
            $dllPath = $commonLocation
        }
        else {
            $dllPath = (Get-ChildItem $vsPath -Recurse -File -Filter Microsoft.VisualStudio.DevShell.dll).FullName
        }
        Import-Module -Force $dllPath
        Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64"
    }
}

if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust is not installed, please install it first."
    Write-Error "https://www.rust-lang.org/tools/install"
    return
}
if (!(Get-Command cbindgen -ErrorAction SilentlyContinue)) {
    Write-Error "cbindgen is not installed, please install it first."
    Write-Error "cargo install cbindgen"
    return
}

cargo build --release
cbindgen --config .\cbindgen.toml --crate onnxrust_proto --output bindings/c/onnxrust_proto.h --lang c
cbindgen --config .\cbindgen.toml --crate onnxrust_proto --output bindings/cpp/onnxrust_proto.hpp --lang c++
cl /nologo /W4 /O2 /Fe: c_test.exe .\bindings\tests\test.c .\target\release\onnxrust_proto_api.dll.lib
cl /nologo /EHsc /std:c++20 /W4 /O2 /Fe: cpp_test.exe .\bindings\tests\test.cpp .\target\release\onnxrust_proto_api.dll.lib
csc /nologo /t:library /unsafe /out:onnxrust_proto_api_cs.dll .\bindings\cs\onnxrust_proto.cs
csc /nologo /t:exe /out:cs_test.exe /r:onnxrust_proto_api_cs.dll .\bindings\tests\test.cs
Copy-Item .\bindings\py\onnxrust_proto.py .
Copy-Item .\bindings\tests\test.py .
Copy-Item .\target\release\onnxrust_proto_api.dll .
Write-Output "Running C test..."
& .\c_test.exe | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "C test failed."
}
else {
    Write-Host "C test passed." -ForegroundColor Green
}
Write-Output "Running C++ test..."
& .\cpp_test.exe | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "C++ test failed."
}
else {
    Write-Host "C++ test passed." -ForegroundColor Green
}
Write-Output "Running Python test..."
& py -3 .\test.py | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python test failed."
}
else {
    Write-Host "Python test passed." -ForegroundColor Green
}
Write-Output "Running C# test..."
& .\cs_test.exe | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "C# test failed."
}
else {
    Write-Host "C# test passed." -ForegroundColor Green
}
Remove-Item .\*.exe
Remove-Item .\*.obj
Remove-Item .\onnxrust_proto.py
Remove-Item .\test.py
Remove-Item .\*.dll
