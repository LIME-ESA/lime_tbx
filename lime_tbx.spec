# -*- mode: python ; coding: utf-8 -*-

import sys

from PyInstaller.utils.hooks import copy_metadata

block_cipher = None


a_pathex = []
a_binaries = []
a_datas = []
a_icon = ''
a_hooksconfig = {}
e_console = False
deploy_name = "LimeTBX"

if sys.platform == 'linux':
    src_path = 'lime_tbx/'
    a_pathex = ['./.venv/lib/python3.9/site-packages/', './.venv/lib64/python3.9/site-packages/']
    a_binaries = [
        ('.venv/lib/python3.9/site-packages/spiceypy/utils/libcspice.so', './spiceypy/utils'),
        (src_path + 'business/eocfi_adapter/eocfi_c/bin/get_positions_linux', './lime_tbx/business/eocfi_adapter/eocfi_c/bin'),
    ]
    a_datas = [
        (src_path + 'presentation/gui/assets/style.qss', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/style_constants.txt', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/lime_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/cropped_lime_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/esa_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/uva_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/goa_uva_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/npl_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/vito_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/spinner.gif', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/NotesEsaBol.otf', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/NotesEsaReg.otf', './lime_tbx/presentation/gui/assets'),
        (src_path + 'business/interpolation/interp_data/assets/SomeMoonReflectances.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/Apollo16.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/Breccia.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/Composite.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/ds_ASD_32.nc', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/spectral_integration/assets/interpolated_model_fwhm_1_1_triangle.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/spectral_integration/assets/interpolated_model_fwhm_3_1_gaussian.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/spectral_integration/assets/asd_fwhm.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/spectral_integration/assets/responses_1088.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/wehrli_asc.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_cimel.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_asd.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_fwhm_3_1_gaussian.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_fwhm_1_1_triangle.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
    ]
    a_icon = src_path + 'presentation/gui/assets/lime_logo.ico'
    a_hooksconfig = {
        "gi": {
            "icons": ["Adwaita"],
            "themes": ["Adwaita"],
        },
    }
elif sys.platform == 'win32' or sys.platform == 'win64':
    src_path = 'lime_tbx\\'
    venv_path = '.venv'
    a_pathex = [f'.\\{venv_path}\\Lib\\site-packages\\']
    a_binaries = [
        (f'{venv_path}\\Lib\\site-packages\\spiceypy\\utils\\libcspice.dll', '.\\spiceypy\\utils'),
        (src_path + 'business\\eocfi_adapter\\eocfi_c\\bin\\get_positions_win64.exe', '.\\lime_tbx\\business\\eocfi_adapter\\eocfi_c\\bin'),
        (src_path + 'business\\eocfi_adapter\\eocfi_c\\bin\\msvcr100.dll', '.\\lime_tbx\\business\\eocfi_adapter\\eocfi_c\\bin'),
        (src_path + 'business\\eocfi_adapter\\eocfi_c\\bin\\pthreadVC2.dll', '.\\lime_tbx\\business\\eocfi_adapter\\eocfi_c\\bin'),
    ]
    a_datas = [
        (src_path + 'presentation\\gui\\assets\\style.qss', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\style_constants.txt', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\lime_logo.png', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\cropped_lime_logo.png', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\esa_logo.png', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\uva_logo.png', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\goa_uva_logo.png', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\npl_logo.png', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\vito_logo.png', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\spinner.gif', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\NotesEsaBol.otf', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'presentation\\gui\\assets\\NotesEsaReg.otf', '.\\lime_tbx\\presentation\\gui\\assets'),
        (src_path + 'business\\interpolation\\interp_data\\assets\\SomeMoonReflectances.txt', '.\\lime_tbx\\business\\interpolation\\interp_data\\assets'),
        (src_path + 'business\\interpolation\\interp_data\\assets\\Apollo16.txt', '.\\lime_tbx\\business\\interpolation\\interp_data\\assets'),
        (src_path + 'business\\interpolation\\interp_data\\assets\\Breccia.txt', '.\\lime_tbx\\business\\interpolation\\interp_data\\assets'),
        (src_path + 'business\\interpolation\\interp_data\\assets\\Composite.txt', '.\\lime_tbx\\business\\interpolation\\interp_data\\assets'),
        (src_path + 'business\\interpolation\\interp_data\\assets\\ds_ASD_32.nc', '.\\lime_tbx\\business\\interpolation\\interp_data\\assets'),
        (src_path + 'business\\spectral_integration\\assets\\interpolated_model_fwhm_1_1_triangle.csv', '.\\lime_tbx\\business\\spectral_integration\\assets'),
        (src_path + 'business\\spectral_integration\\assets\\interpolated_model_fwhm_3_1_gaussian.csv', '.\\lime_tbx\\business\\spectral_integration\\assets'),
        (src_path + 'business\\spectral_integration\\assets\\asd_fwhm.csv', '.\\lime_tbx\\business\\spectral_integration\\assets'),
        (src_path + 'business\\spectral_integration\\assets\\responses_1088.csv', '.\\lime_tbx\\business\\spectral_integration\\assets'),
        (src_path + 'business\\lime_algorithms\\lime\\assets\\wehrli_asc.csv', '.\\lime_tbx\\business\\lime_algorithms\\lime\\assets'),
        (src_path + 'business\\lime_algorithms\\lime\\assets\\tsis_cimel.csv', '.\\lime_tbx\\business\\lime_algorithms\\lime\\assets'),
        (src_path + 'business\\lime_algorithms\\lime\\assets\\tsis_asd.csv', '.\\lime_tbx\\business\\lime_algorithms\\lime\\assets'),
        (src_path + 'business\\lime_algorithms\\lime\\assets\\tsis_fwhm_3_1_gaussian.csv', '.\\lime_tbx\\business\\lime_algorithms\\lime\\assets'),
        (src_path + 'business\\lime_algorithms\\lime\\assets\\tsis_fwhm_1_1_triangle.csv', '.\\lime_tbx\\business\\lime_algorithms\\lime\\assets'),
    ] + copy_metadata("lime_tbx")
    a_icon = src_path + 'presentation\\gui\\assets\\lime_logo.ico'
    e_console = True
elif sys.platform == 'darwin':
    eocfi_bin_path = 'business/eocfi_adapter/eocfi_c/bin/get_positions_darwin'
    import platform
    if "ARM" in platform.version().upper():
        eocfi_bin_path = 'business/eocfi_adapter/eocfi_c/bin/get_positions_darwin_arm'
    src_path = 'lime_tbx/'
    a_pathex = ['./.venv/lib/python3.9/site-packages/', './.venv/lib64/python3.9/site-packages/']
    a_binaries = [
        ('.venv/lib/python3.9/site-packages/spiceypy/utils/libcspice.so', './spiceypy/utils'),
        (src_path + eocfi_bin_path, './lime_tbx/business/eocfi_adapter/eocfi_c/bin'),
    ]
    a_datas = [
        (src_path + 'presentation/gui/assets/style.qss', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/style_constants.txt', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/style_constants_darwin.txt', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/lime_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/cropped_lime_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/esa_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/uva_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/goa_uva_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/npl_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/vito_logo.png', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/spinner.gif', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/NotesEsaBol.otf', './lime_tbx/presentation/gui/assets'),
        (src_path + 'presentation/gui/assets/NotesEsaReg.otf', './lime_tbx/presentation/gui/assets'),
        (src_path + 'business/interpolation/interp_data/assets/SomeMoonReflectances.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/Apollo16.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/Breccia.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/Composite.txt', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/interpolation/interp_data/assets/ds_ASD_32.nc', './lime_tbx/business/interpolation/interp_data/assets'),
        (src_path + 'business/spectral_integration/assets/interpolated_model_fwhm_1_1_triangle.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/spectral_integration/assets/interpolated_model_fwhm_3_1_gaussian.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/spectral_integration/assets/asd_fwhm.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/spectral_integration/assets/responses_1088.csv', './lime_tbx/business/spectral_integration/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/wehrli_asc.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_cimel.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_asd.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_fwhm_3_1_gaussian.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        (src_path + 'business/lime_algorithms/lime/assets/tsis_fwhm_1_1_triangle.csv', './lime_tbx/business/lime_algorithms/lime/assets'),
        ('lime_tbx.egg-info', 'lime_tbx.egg-info'),
    ]
    a_icon = src_path + 'presentation/gui/assets/lime_logo.icns'
runner_file = src_path + '__main__.py'

a = Analysis(
    [runner_file],
    pathex=a_pathex,
    binaries=a_binaries,
    datas=a_datas,
    hiddenimports=["sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting", "sklearn.utils._vector_sentinel", "jaraco"],
    hookspath=[],
    hooksconfig=a_hooksconfig,
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=deploy_name + ".exe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=e_console,
    icon=a_icon,
    disable_windowed_traceback=False,
    target_arch=None,
    argv_emulation=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=deploy_name,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=deploy_name+'.app',
        icon=a_icon,
        bundle_identifier="int.esa.LimeTBX",
    )
