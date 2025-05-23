/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Serapheim Dimitropoulos @ Weka. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include "weka_utils.h"

nixl_status_t wekaUtil::registerFileHandle(int fd,
                                          size_t size,
                                          std::string metaInfo,
                                          wekaFileHandle& weka_handle)
{
    CUfileError_t status;
    CUfileDescr_t descr;
    CUfileHandle_t handle;

    descr = {};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "file register error:" << std::endl;
        return NIXL_ERR_BACKEND;
    }

    weka_handle.cu_fhandle = handle;
    weka_handle.fd = fd;
    weka_handle.size = size;
    weka_handle.metadata = metaInfo;

    return NIXL_SUCCESS;
}

nixl_status_t wekaUtil::registerBufHandle(void *ptr,
                                         size_t size,
                                         int flags)
{
    CUfileError_t status;

    status = cuFileBufRegister(ptr, size, flags);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "Warning: Buffer registration failed - will use compat mode\n";
    }
    return NIXL_SUCCESS;
}

nixl_status_t wekaUtil::openWekaDriver()
{
    CUfileError_t err;

    err = cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        std::cerr << "Error initializing GPU Direct Storage driver\n";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

void wekaUtil::closeWekaDriver()
{
    cuFileDriverClose();
}

void wekaUtil::deregisterFileHandle(wekaFileHandle& handle)
{
    cuFileHandleDeregister(handle.cu_fhandle);
}

nixl_status_t wekaUtil::deregisterBufHandle(void *ptr)
{
    CUfileError_t status;

    status = cuFileBufDeregister(ptr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "Error De-Registering Buffer\n";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}
