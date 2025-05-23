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
#ifndef __WEKA_UTILS_H
#define __WEKA_UTILS_H

// TODO(Serapheim): Check unneeded imports
#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include <cufile.h>

class wekaFileHandle {
    public:
        int fd;
        size_t size;
        std::string metadata;
        CUfileHandle_t cu_fhandle;
};

class wekaMemBuf {
    public:
        void *base;
        size_t size;
};

class wekaUtil {
    public:
        wekaUtil() {}
        ~wekaUtil() {}
        nixl_status_t registerFileHandle(int fd, size_t size,
                                       std::string metaInfo,
                                       wekaFileHandle& handle);
        nixl_status_t registerBufHandle(void *ptr, size_t size, int flags);
        void deregisterFileHandle(wekaFileHandle& handle);
        nixl_status_t deregisterBufHandle(void *ptr);
        nixl_status_t openWekaDriver();
        void closeWekaDriver();
};
#endif
