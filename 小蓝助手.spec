# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    datas=[('data/knowledge.xlsx', 'data'), ('audio', 'audio')],
    hiddenimports=['webrtcvad', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree'],
    hookspath=['extra-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'qtpy', 'setuptools', 'pkg_resources'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='小蓝助手',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['robot.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='小蓝助手',
)
