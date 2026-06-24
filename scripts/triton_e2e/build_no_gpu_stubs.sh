#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${1:-/home/sunrise/triton_backend_BPU/tritonserver_dist}"
STUB_DIR="$DIST_DIR/stubs"
mkdir -p "$STUB_DIR"

cat > "$STUB_DIR/cudart_stub.c" <<'STUB'
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaHostFn_t;

enum {
  cudaSuccess = 0,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidDevice = 10,
  cudaErrorNotSupported = 801
};

const char* cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorNoDevice: return "no CUDA-capable device is detected";
    case cudaErrorInvalidDevice: return "invalid device ordinal";
    case cudaErrorNotSupported: return "operation not supported by no-GPU stub";
    default: return "CUDA unavailable on this RDK S100P no-GPU stub";
  }
}

cudaError_t cudaGetDeviceCount(int* count) { if (count) *count = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int device) { (void)device; return cudaErrorInvalidDevice; }
cudaError_t cudaGetDevice(int* device) { if (device) *device = 0; return cudaErrorNoDevice; }
cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) { if (free_b) *free_b = 0; if (total_b) *total_b = 0; return cudaErrorNoDevice; }
cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) { (void)device; if (pciBusId && len > 0) pciBusId[0] = '\0'; return cudaErrorNoDevice; }
cudaError_t cudaGetDeviceProperties(void* prop, int device) { (void)prop; (void)device; return cudaErrorNoDevice; }
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) { (void)device; (void)peerDevice; if (canAccessPeer) *canAccessPeer = 0; return cudaSuccess; }
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) { (void)peerDevice; (void)flags; return cudaErrorNoDevice; }
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) { if (leastPriority) *leastPriority = 0; if (greatestPriority) *greatestPriority = 0; return cudaSuccess; }
cudaError_t cudaMalloc(void** devPtr, size_t size) { (void)size; if (devPtr) *devPtr = 0; return cudaErrorNoDevice; }
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) { (void)size; (void)flags; if (devPtr) *devPtr = 0; return cudaErrorNoDevice; }
cudaError_t cudaFree(void* devPtr) { (void)devPtr; return cudaSuccess; }
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) { (void)size; (void)flags; if (pHost) *pHost = 0; return cudaErrorNoDevice; }
cudaError_t cudaFreeHost(void* ptr) { (void)ptr; return cudaSuccess; }
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) { (void)kind; if (dst && src && count) memcpy(dst, src, count); return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, cudaStream_t stream) { (void)stream; return cudaMemcpy(dst, src, count, kind); }
cudaError_t cudaMemset(void* devPtr, int value, size_t count) { if (devPtr && count) memset(devPtr, value, count); return cudaSuccess; }
cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) { (void)devPtr; (void)count; (void)dstDevice; (void)stream; return cudaErrorNoDevice; }
cudaError_t cudaStreamCreate(cudaStream_t* pStream) { if (pStream) *pStream = 0; return cudaSuccess; }
cudaError_t cudaStreamDestroy(cudaStream_t stream) { (void)stream; return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; return cudaSuccess; }
cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) { (void)hStream; if (flags) *flags = 0; return cudaSuccess; }
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) { (void)stream; if (fn) ((void (*)(void*))fn)(userData); return cudaSuccess; }
cudaError_t cudaIpcOpenMemHandle(void** devPtr, void* handle, unsigned int flags) { (void)handle; (void)flags; if (devPtr) *devPtr = 0; return cudaErrorNoDevice; }
cudaError_t cudaIpcCloseMemHandle(void* devPtr) { (void)devPtr; return cudaSuccess; }
cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, int driverStatus) { (void)symbol; (void)flags; (void)driverStatus; if (funcPtr) *funcPtr = 0; return cudaErrorNoDevice; }
STUB

cat > "$STUB_DIR/dcgm_stub.c" <<'STUB'
#include <stdint.h>

typedef int dcgmReturn_t;
typedef int dcgmHandle_t;
typedef int dcgmGpuGrp_t;
typedef int dcgmFieldGrp_t;

enum { DCGM_ST_OK = 0, DCGM_ST_NO_DATA = -10, DCGM_ST_NOT_SUPPORTED = -25 };

dcgmReturn_t dcgmInit(void) { return DCGM_ST_OK; }
dcgmReturn_t dcgmShutdown(void) { return DCGM_ST_OK; }
dcgmReturn_t dcgmStartEmbedded(int opMode, dcgmHandle_t* pDcgmHandle) { (void)opMode; if (pDcgmHandle) *pDcgmHandle = 0; return DCGM_ST_OK; }
dcgmReturn_t dcgmStopEmbedded(dcgmHandle_t pDcgmHandle) { (void)pDcgmHandle; return DCGM_ST_OK; }
dcgmReturn_t dcgmConnect(const char* ipAddress, dcgmHandle_t* pDcgmHandle) { (void)ipAddress; if (pDcgmHandle) *pDcgmHandle = 0; return DCGM_ST_OK; }
dcgmReturn_t dcgmDisconnect(dcgmHandle_t pDcgmHandle) { (void)pDcgmHandle; return DCGM_ST_OK; }
dcgmReturn_t dcgmGetAllDevices(dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[], int* count) { (void)pDcgmHandle; (void)gpuIdList; if (count) *count = 0; return DCGM_ST_OK; }
dcgmReturn_t dcgmGroupCreate(dcgmHandle_t pDcgmHandle, int type, char* groupName, dcgmGpuGrp_t* pDcgmGrpId) { (void)pDcgmHandle; (void)type; (void)groupName; if (pDcgmGrpId) *pDcgmGrpId = 0; return DCGM_ST_OK; }
dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId) { (void)pDcgmHandle; (void)groupId; return DCGM_ST_OK; }
dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t pDcgmHandle, int numFieldIds, unsigned short* fieldIds, char* fieldGroupName, dcgmFieldGrp_t* dcgmFieldGroupId) { (void)pDcgmHandle; (void)numFieldIds; (void)fieldIds; (void)fieldGroupName; if (dcgmFieldGroupId) *dcgmFieldGroupId = 0; return DCGM_ST_OK; }
dcgmReturn_t dcgmWatchFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId, long long updateFreq, double maxKeepAge, int maxKeepSamples) { (void)pDcgmHandle; (void)groupId; (void)fieldGroupId; (void)updateFreq; (void)maxKeepAge; (void)maxKeepSamples; return DCGM_ST_OK; }
dcgmReturn_t dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate) { (void)pDcgmHandle; (void)waitForUpdate; return DCGM_ST_OK; }
dcgmReturn_t dcgmGetLatestValuesForFields(dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldIds[], unsigned int count, void* values) { (void)pDcgmHandle; (void)gpuId; (void)fieldIds; (void)count; (void)values; return DCGM_ST_NO_DATA; }
dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle, int gpuId, void* pDcgmAttr) { (void)pDcgmHandle; (void)gpuId; (void)pDcgmAttr; return DCGM_ST_NOT_SUPPORTED; }
STUB

cc -shared -fPIC "$STUB_DIR/cudart_stub.c" -Wl,-soname,libcudart.so.13 -o "$STUB_DIR/libcudart.so.13"
cc -shared -fPIC "$STUB_DIR/dcgm_stub.c" -Wl,-soname,libdcgm.so.4 -o "$STUB_DIR/libdcgm.so.4"
ls -l "$STUB_DIR/libcudart.so.13" "$STUB_DIR/libdcgm.so.4"
