# -*- mode: python ; coding: utf-8 -*-

import sys

block_cipher = None


a_pathex = []
a_binaries = []
a_datas = []
a_icon = ''
e_console = False

if sys.platform == 'linux':
    src_path = 'lime_tbx/'
    a_pathex = ['./.venv/lib/python3.8/site-packages/', './.venv/lib64/python3.8/site-packages/']
    a_binaries = [
        ('.venv/lib/python3.8/site-packages/spiceypy/utils/libcspice.so', './spiceypy/utils'),
        (src_path + 'eocfi_adapter/eocfi_c/bin/get_positions_linux.so', './lime_tbx/eocfi_adapter/eocfi_c/bin'),
        (src_path + 'eocfi_adapter/eocfi_c/bin/get_positions_linux', './lime_tbx/eocfi_adapter/eocfi_c/bin'),
    ]
    a_datas = [
        (src_path + 'gui/assets/style.qss', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/style_constants.txt', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/lime_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/spinner.gif', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/NotesEsaBol.otf', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/NotesEsaReg.otf', './lime_tbx/gui/assets'),
        (src_path + 'coefficients/access_data/assets/coefficients.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/coefficients_cimel.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/u_coefficients_cimel.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'interpolation/access_data/assets/SomeMoonReflectances.txt', './lime_tbx/interpolation/access_data/assets'),
        (src_path + 'lime_algorithms/rolo/assets/wehrli_asc.csv', './lime_tbx/lime_algorithms/rolo/assets'),
    ]
    a_icon = src_path + 'gui/assets/lime_logo.ico'
elif sys.platform == 'win32' or sys.platform == 'win64':
    src_path = 'lime_tbx\\'
    a_pathex = ['.\\.venv\\Lib\\site-packages\\']
    a_binaries = [
        ('.venv\\Lib\\site-packages\\spiceypy\\utils\\libcspice.dll', '.\\spiceypy\\utils'),
        (src_path + 'eocfi_adapter\\eocfi_c\\bin\\get_positions_win64.dll', '.\\lime_tbx\\eocfi_adapter\\eocfi_c\\bin'),
        (src_path + 'eocfi_adapter\\eocfi_c\\bin\\get_positions_win64.exe', '.\\lime_tbx\\eocfi_adapter\\eocfi_c\\bin'),
        (src_path + 'eocfi_adapter\\eocfi_c\\bin\\msvcr100.dll', '.\\lime_tbx\\eocfi_adapter\\eocfi_c\\bin'),
        (src_path + 'eocfi_adapter\\eocfi_c\\bin\\pthreadVC2.dll', '.\\lime_tbx\\eocfi_adapter\\eocfi_c\\bin'),
    ]
    a_datas = [
        (src_path + 'gui\\assets\\style.qss', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\style_constants.txt', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\lime_logo.png', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\spinner.gif', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\NotesEsaBol.otf', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\NotesEsaReg.otf', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'coefficients\\access_data\\assets\\coefficients.csv', '.\\lime_tbx\\coefficients\\access_data\\assets'),
        (src_path + 'coefficients\\access_data\\assets\\coefficients_cimel.csv', '.\\lime_tbx\\coefficients\\access_data\\assets'),
        (src_path + 'coefficients\\access_data\\assets\\u_coefficients_cimel.csv', '.\\lime_tbx\\coefficients\\access_data\\assets'),
        (src_path + 'interpolation\\access_data\\assets\\SomeMoonReflectances.txt', '.\\lime_tbx\\interpolation\\access_data\\assets'),
        (src_path + 'lime_algorithms\\rolo\\assets\\wehrli_asc.csv', '.\\lime_tbx\\lime_algorithms\\rolo\\assets'),
    ]
    a_icon = src_path + 'gui\\assets\\lime_logo.ico'
    e_console = True
runner_file = src_path + 'main.py'

a = Analysis([runner_file],
            pathex=a_pathex,
            binaries=a_binaries,
            datas=a_datas,
            hiddenimports=["sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting", "sklearn.utils._vector_sentinel"],
            hookspath=[],
            hooksconfig={},
            runtime_hooks=[],
            excludes=[],
            win_no_prefer_redirects=False,
            win_private_assemblies=False,
            cipher=block_cipher,
            noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

if sys.platform == 'win32' or sys.platform == 'win64' or sys.platform == 'linux':
    exe = EXE(pyz,
            a.scripts,
            a.binaries,
            a.zipfiles,
            a.datas,  
            [],
            name='LIME_TBX',
            debug=False,
            bootloader_ignore_signals=False,
            strip=False,
            upx=True,
            upx_exclude=[],
            runtime_tmpdir=None,
            console=e_console,
            icon=a_icon,
            disable_windowed_traceback=False,
            target_arch=None,
            codesign_identity=None,
            entitlements_file=None )

