param([string]$Kotlinc = 'kotlinc')
$ErrorActionPreference = 'Stop'
& $Kotlinc (Join-Path $PSScriptRoot 'src/main/kotlin/Main.kt') -jvm-target 17 -include-runtime -d (Join-Path $PSScriptRoot 'lab02.jar')
if ($LASTEXITCODE -ne 0) { throw 'Сборка Kotlin завершилась с ошибкой' }
