$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
chcp 65001 | Out-Null

if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSDefaultParameterValues["Out-File:Encoding"] = "utf8NoBOM"
  $PSDefaultParameterValues["Set-Content:Encoding"] = "utf8NoBOM"
  $PSDefaultParameterValues["Add-Content:Encoding"] = "utf8NoBOM"
} else {
  $PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
  $PSDefaultParameterValues["Set-Content:Encoding"] = "utf8"
  $PSDefaultParameterValues["Add-Content:Encoding"] = "utf8"
}
