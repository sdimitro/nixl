# SPDX-FileCopyrightText: Copyright (c) 2025 Serapheim Dimitropoulos @ Weka. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

gds_path = get_option('gds_path')
if gds_path != ''
    gds_lib_path = gds_path + '/lib'
    gds_inc_path = gds_path + '/include'
    cufile_dep = declare_dependency(
        link_args: ['-L' + gds_lib_path, '-lcufile'],
        include_directories: include_directories(gds_inc_path))
endif

if 'WEKA' in static_plugins
    weka_backend_lib = static_library('WEKA',
        'weka_utils.cpp', 'weka_utils.h',
        'weka_backend.cpp', 'weka_backend.h',
        'weka_plugin.cpp',
        dependencies: [nixl_infra, nixl_common_dep, cuda_dep, cufile_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        cpp_args: compile_flags,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    weka_backend_lib = shared_library('WEKA',
        'weka_utils.cpp', 'weka_utils.h',
        'weka_backend.cpp', 'weka_backend.h',
        'weka_plugin.cpp',
        dependencies: [nixl_infra, nixl_common_dep, cuda_dep, cufile_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        cpp_args: ['-fPIC'],
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir)
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "WEKA=' + weka_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

weka_backend_interface = declare_dependency(link_with: weka_backend_lib)
