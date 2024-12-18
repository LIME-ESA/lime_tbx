# -*- mode: python ; coding: utf-8 -*-

import sys

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
    a_pathex = ['./.venv/lib/python3.8/site-packages/', './.venv/lib64/python3.8/site-packages/']
    a_binaries = [
        ('.venv/lib/python3.8/site-packages/spiceypy/utils/libcspice.so', './spiceypy/utils'),
        (src_path + 'eocfi_adapter/eocfi_c/bin/get_positions_linux', './lime_tbx/eocfi_adapter/eocfi_c/bin'),
    ]
    a_datas = [
        (src_path + 'gui/assets/style.qss', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/style_constants.txt', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/lime_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/esa_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/uva_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/goa_uva_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/npl_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/vito_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/spinner.gif', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/NotesEsaBol.otf', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/NotesEsaReg.otf', './lime_tbx/gui/assets'),
        (src_path + 'coefficients/access_data/assets/coefficients.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/ds_cimel_coeff.nc', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/coefficients_cimel.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/u_coefficients_cimel.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'interpolation/interp_data/assets/SomeMoonReflectances.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/Apollo16.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/Breccia.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/Composite.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/ds_ASD.nc', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'spectral_integration/assets/interpolated_model_fwhm_1_1_triangle.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'spectral_integration/assets/interpolated_model_fwhm_3_1_gaussian.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'spectral_integration/assets/asd_fwhm.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'spectral_integration/assets/responses_1088.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'lime_algorithms/lime/assets/wehrli_asc.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_cimel.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_asd.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_fwhm_3_1_gaussian.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_fwhm_1_1_triangle.csv', './lime_tbx/lime_algorithms/lime/assets'),
    ]
    a_icon = src_path + 'gui/assets/lime_logo.ico'
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
        (src_path + 'eocfi_adapter\\eocfi_c\\bin\\get_positions_win64.exe', '.\\lime_tbx\\eocfi_adapter\\eocfi_c\\bin'),
        (src_path + 'eocfi_adapter\\eocfi_c\\bin\\msvcr100.dll', '.\\lime_tbx\\eocfi_adapter\\eocfi_c\\bin'),
        (src_path + 'eocfi_adapter\\eocfi_c\\bin\\pthreadVC2.dll', '.\\lime_tbx\\eocfi_adapter\\eocfi_c\\bin'),
        (f'{venv_path}\\Lib\\site-packages\\shiboken2\\files.dir', '.\\shiboken2\\files.dir'),
    ]
    a_datas = [
        (src_path + 'gui\\assets\\style.qss', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\style_constants.txt', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\lime_logo.png', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\esa_logo.png', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\uva_logo.png', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\goa_uva_logo.png', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\npl_logo.png', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\vito_logo.png', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\spinner.gif', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\NotesEsaBol.otf', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'gui\\assets\\NotesEsaReg.otf', '.\\lime_tbx\\gui\\assets'),
        (src_path + 'coefficients\\access_data\\assets\\coefficients.csv', '.\\lime_tbx\\coefficients\\access_data\\assets'),
        (src_path + 'coefficients\\access_data\\assets\\ds_cimel_coeff.nc', '.\\lime_tbx\\coefficients\\access_data\\assets'),
        (src_path + 'coefficients\\access_data\\assets\\coefficients_cimel.csv', '.\\lime_tbx\\coefficients\\access_data\\assets'),
        (src_path + 'coefficients\\access_data\\assets\\u_coefficients_cimel.csv', '.\\lime_tbx\\coefficients\\access_data\\assets'),
        (src_path + 'interpolation\\interp_data\\assets\\SomeMoonReflectances.txt', '.\\lime_tbx\\interpolation\\interp_data\\assets'),
        (src_path + 'interpolation\\interp_data\\assets\\Apollo16.txt', '.\\lime_tbx\\interpolation\\interp_data\\assets'),
        (src_path + 'interpolation\\interp_data\\assets\\Breccia.txt', '.\\lime_tbx\\interpolation\\interp_data\\assets'),
        (src_path + 'interpolation\\interp_data\\assets\\Composite.txt', '.\\lime_tbx\\interpolation\\interp_data\\assets'),
        (src_path + 'interpolation\\interp_data\\assets\\ds_ASD.nc', '.\\lime_tbx\\interpolation\\interp_data\\assets'),
        (src_path + 'spectral_integration\\assets\\interpolated_model_fwhm_1_1_triangle.csv', '.\\lime_tbx\\spectral_integration\\assets'),
        (src_path + 'spectral_integration\\assets\\interpolated_model_fwhm_3_1_gaussian.csv', '.\\lime_tbx\\spectral_integration\\assets'),
        (src_path + 'spectral_integration\\assets\\asd_fwhm.csv', '.\\lime_tbx\\spectral_integration\\assets'),
        (src_path + 'spectral_integration\\assets\\responses_1088.csv', '.\\lime_tbx\\spectral_integration\\assets'),
        (src_path + 'lime_algorithms\\lime\\assets\\wehrli_asc.csv', '.\\lime_tbx\\lime_algorithms\\lime\\assets'),
        (src_path + 'lime_algorithms\\lime\\assets\\tsis_cimel.csv', '.\\lime_tbx\\lime_algorithms\\lime\\assets'),
        (src_path + 'lime_algorithms\\lime\\assets\\tsis_asd.csv', '.\\lime_tbx\\lime_algorithms\\lime\\assets'),
        (src_path + 'lime_algorithms\\lime\\assets\\tsis_fwhm_3_1_gaussian.csv', '.\\lime_tbx\\lime_algorithms\\lime\\assets'),
        (src_path + 'lime_algorithms\\lime\\assets\\tsis_fwhm_1_1_triangle.csv', '.\\lime_tbx\\lime_algorithms\\lime\\assets'),
    ]
    a_icon = src_path + 'gui\\assets\\lime_logo.ico'
    e_console = True
elif sys.platform == 'darwin':
    eocfi_bin_path = 'eocfi_adapter/eocfi_c/bin/get_positions_darwin'
    import platform
    if "ARM" in platform.version().upper():
        eocfi_bin_path = 'eocfi_adapter/eocfi_c/bin/get_positions_darwin_arm'
    src_path = 'lime_tbx/'
    a_pathex = ['./.venv/lib/python3.9/site-packages/', './.venv/lib64/python3.9/site-packages/']
    a_binaries = [
        ('.venv/lib/python3.9/site-packages/spiceypy/utils/libcspice.so', './spiceypy/utils'),
        (src_path + eocfi_bin_path, './lime_tbx/eocfi_adapter/eocfi_c/bin'),
    ]
    a_datas = [
        (src_path + 'gui/assets/style.qss', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/style_constants.txt', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/style_constants_darwin.txt', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/lime_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/esa_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/uva_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/goa_uva_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/npl_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/vito_logo.png', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/spinner.gif', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/NotesEsaBol.otf', './lime_tbx/gui/assets'),
        (src_path + 'gui/assets/NotesEsaReg.otf', './lime_tbx/gui/assets'),
        (src_path + 'coefficients/access_data/assets/coefficients.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/ds_cimel_coeff.nc', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/coefficients_cimel.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'coefficients/access_data/assets/u_coefficients_cimel.csv', './lime_tbx/coefficients/access_data/assets'),
        (src_path + 'interpolation/interp_data/assets/SomeMoonReflectances.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/Apollo16.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/Breccia.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/Composite.txt', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'interpolation/interp_data/assets/ds_ASD.nc', './lime_tbx/interpolation/interp_data/assets'),
        (src_path + 'spectral_integration/assets/interpolated_model_fwhm_1_1_triangle.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'spectral_integration/assets/interpolated_model_fwhm_3_1_gaussian.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'spectral_integration/assets/asd_fwhm.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'spectral_integration/assets/responses_1088.csv', './lime_tbx/spectral_integration/assets'),
        (src_path + 'lime_algorithms/lime/assets/wehrli_asc.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_cimel.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_asd.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_fwhm_3_1_gaussian.csv', './lime_tbx/lime_algorithms/lime/assets'),
        (src_path + 'lime_algorithms/lime/assets/tsis_fwhm_1_1_triangle.csv', './lime_tbx/lime_algorithms/lime/assets'),
    ]
    a_icon = src_path + 'gui/assets/lime_logo.icns'
runner_file = src_path + 'main.py'

a = Analysis(
    [runner_file],
    pathex=a_pathex,
    binaries=a_binaries,
    datas=a_datas,
    hiddenimports=["sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting", "sklearn.utils._vector_sentinel"],
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
