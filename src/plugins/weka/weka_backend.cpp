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
#include <cassert>
#include <iostream>
#include <cufile.h>
#include <thread>
#include "weka_backend.h"
#include "common/str_tools.h"

/** Setting the max request size to 64 MB */
#define DEFAULT_MAX_REQUEST_SIZE (64 * 1024 * 1024)  // 64MB

nixlWekaEngine::nixlWekaEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params)
{
    weka_utils = new wekaUtil();

    max_request_size = DEFAULT_MAX_REQUEST_SIZE;

    // Read custom parameters if available
    nixl_b_params_t* custom_params = init_params->customParams;
    if (custom_params) {
        // Configure max_request_size
        if (custom_params->count("max_request_size") > 0) {
            try {
                max_request_size = std::stoul((*custom_params)["max_request_size"]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid max_request_size parameter: " << e.what() << std::endl;
                this->initErr = true;
                return;
            }
        }
        // TODO(Serapheim): configure the thread pool
    }

    this->initErr = false;
    if (weka_utils->openWekaDriver() == NIXL_ERR_BACKEND) {
        this->initErr = true;
        return;
    }
}

nixl_status_t nixlWekaEngine::registerMem(const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    nixl_status_t status = NIXL_SUCCESS;
    nixlWekaMetadata *md = new nixlWekaMetadata();
    md->type = nixl_mem;
    cudaError_t error_id;

    switch (nixl_mem) {
        case FILE_SEG: {
            auto it = weka_file_map.find(mem.devId);
            if (it != weka_file_map.end()) {
                // no need to re-register
                md->handle = it->second;
                md->handle.size = mem.len;
                md->handle.metadata = mem.metaInfo;
                break;
            }

            status = weka_utils->registerFileHandle(mem.devId, mem.len,
                                                    mem.metaInfo, md->handle);
            if (status == NIXL_SUCCESS) {
                weka_file_map[mem.devId] = md->handle;
            }
            break;
        }

        case VRAM_SEG: {
            error_id = cudaSetDevice(mem.devId);
            if (error_id != cudaSuccess) {
                std::cerr << "cudaSetDevice returned " << cudaGetErrorString(error_id)
                          << " for device ID " << mem.devId << std::endl;
                delete md;
                return NIXL_ERR_BACKEND;
            }
            status = weka_utils->registerBufHandle((void *)mem.addr, mem.len, 0);
            if (status == NIXL_SUCCESS) {
                md->buf.base = (void *)mem.addr;
                md->buf.size = mem.len;
            }
            break;
        }

        case DRAM_SEG: {
            status = weka_utils->registerBufHandle((void *)mem.addr, mem.len, 0);
            if (status == NIXL_SUCCESS) {
                md->buf.base = (void *)mem.addr;
                md->buf.size = mem.len;
            }
            break;
        }

        default:
            status = NIXL_ERR_BACKEND;
            break;
    }

    if (status != NIXL_SUCCESS) {
        delete md;
        return status;
    }

    out = (nixlBackendMD*)md;
    return status;
}

nixl_status_t nixlWekaEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlWekaMetadata *md = (nixlWekaMetadata *)meta;
    if (md->type == FILE_SEG) {
        weka_utils->deregisterFileHandle(md->handle);
	    weka_file_map.erase(md->handle.fd);
    } else {
        weka_utils->deregisterBufHandle(md->buf.base);
    }
    delete md;
    return NIXL_SUCCESS;
}

nixl_status_t nixlWekaEngine::prepXfer (const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args) const
{
    nixlWekaBackendReqH* weka_handle = new nixlWekaBackendReqH();
    size_t buf_cnt = local.descCount();
    size_t file_cnt = remote.descCount();

    // Basic validation
    if ((buf_cnt != file_cnt) ||
        ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        std::cerr << "Error in count or operation selection\n";
        delete weka_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        std::cerr << "Only support I/O between memory (DRAM/VRAM) and file type\n";
        delete weka_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Clear any existing requests before populating
    weka_handle->request_list.clear();

    // Determine if local is the file segment
    bool is_local_file = (local.getType() == FILE_SEG);

    // Create list of all transfer requests
    for (size_t i = 0; i < buf_cnt; i++) {
        void* base_addr;
        size_t total_size;
        size_t base_offset;
        wekaFileHandle fh;

        // Get transfer parameters based on whether local is file or memory
        if (is_local_file) {
            base_addr = (void*)remote[i].addr;
            if (!base_addr) {
                delete weka_handle;
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = remote[i].len;
            base_offset = (size_t)local[i].addr;

            auto it = weka_file_map.find(local[i].devId);
            if (it == weka_file_map.end()) {
                std::cerr << "File handle not found\n";
                delete weka_handle;
                return NIXL_ERR_NOT_FOUND;
            }
            fh = it->second;
        } else {
            base_addr = (void*)local[i].addr;
            if (!base_addr) {
                delete weka_handle;
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = local[i].len;
            base_offset = (size_t)remote[i].addr;

            auto it = weka_file_map.find(remote[i].devId);
            if (it == weka_file_map.end()) {
                std::cerr << "File handle not found\n";
                delete weka_handle;
                return NIXL_ERR_NOT_FOUND;
            }
            fh = it->second;
        }

        // Split large transfers into multiple requests
        size_t remaining_size = total_size;
        size_t current_offset = 0;

        while (remaining_size > 0) {
            size_t request_size = std::min(remaining_size,
                                       (size_t)max_request_size);

            WekaTransferRequestH req;
            req.addr = (char*)base_addr + current_offset;
            req.size = request_size;
            req.file_offset = base_offset + current_offset;
            req.fh = fh.cu_fhandle;
            req.op = (operation == NIXL_READ) ? CUFILE_READ : CUFILE_WRITE;

            weka_handle->request_list.push_back(req);

            remaining_size -= request_size;
            current_offset += request_size;
        }
    }

    if (weka_handle->request_list.empty()) {
        delete weka_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    weka_handle->needs_prep = false;
    handle = weka_handle;
    return NIXL_SUCCESS;
}

enum op_result_t {
    CUFILE_SUCCESS = 0,
    CUFILE_ERR_OP_FAILED = -1,
    CUFILE_ERR_OP_UNKNOWN= -2,
    CUFILE_ERR_OP_SHORT = -3,
};

int cufile_operation(WekaTransferRequestH* req) {
    ssize_t nbytes = 0;
    if (req->op == CUFILE_READ) {
        nbytes = cuFileRead(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            perror("cuFileRead failed");
            return CUFILE_ERR_OP_FAILED;
        }
    } else if (req->op == CUFILE_WRITE) {
        nbytes = cuFileWrite(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            perror("cuFileWrite failed");
            return CUFILE_ERR_OP_FAILED;
        }
    } else {
        return CUFILE_ERR_OP_UNKNOWN;
    }
    if ((size_t)nbytes != req->size) {
        std::cerr << "error: short " << ((req->op == CUFILE_READ) ? "read: " : "write: ")
                  << nbytes << " out of " << req->size << "bytes - address "
                  << req->addr << std::endl;
        return CUFILE_ERR_OP_SHORT;
    }
    return CUFILE_SUCCESS;
}

nixl_status_t nixlWekaEngine::postXfer(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    nixlWekaBackendReqH* weka_handle = (nixlWekaBackendReqH*)handle;

    if (weka_handle->request_list.empty()) {
        std::cerr << "Empty request list" << std::endl;
        return NIXL_ERR_INVALID_PARAM;
    }

    for (WekaTransferRequestH& req : weka_handle->request_list) {
        weka_handle->io_list.push_back(std::async(std::launch::async, cufile_operation, &req));
    }

    return NIXL_IN_PROG;
}

nixl_status_t nixlWekaEngine::checkXfer(nixlBackendReqH* handle) const
{
    nixlWekaBackendReqH *weka_handle = (nixlWekaBackendReqH *)handle;
    nixl_status_t status = NIXL_SUCCESS;

    for (auto it = weka_handle->io_list.begin(); it != weka_handle->io_list.end(); /* empty */) {
        if (it->wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            return NIXL_IN_PROG;
        }
        int ret = it->get();
        it = weka_handle->io_list.erase(it);
        if (ret != 0) {
            status = NIXL_ERR_UNKNOWN;
            break;
        }
    }

    weka_handle->needs_prep = true;
    return status;
}

nixl_status_t nixlWekaEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlWekaBackendReqH *weka_handle = (nixlWekaBackendReqH *) handle;
    delete weka_handle;
    weka_handle = nullptr;
    return NIXL_SUCCESS;
}

nixlWekaEngine::~nixlWekaEngine() {
    if (weka_utils) {
        weka_utils->closeWekaDriver();
        delete weka_utils;
    }
}
