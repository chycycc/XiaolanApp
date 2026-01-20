from PyInstaller.utils.hooks import collect_dynamic_libs

# 收集 webrtcvad 的 .pyd/.dll（Windows 编译扩展需要）
binaries = collect_dynamic_libs("webrtcvad")
