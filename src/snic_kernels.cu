# include <stdio.h>
# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include <cuda.h>
# include "snic_kernels.cuh"
# include <cooperative_groups.h>
# include <assert.h>
# include <stdint.h>
//How many elements to process at once per heap/threadblock, Depending on the image various number may cause overflow from shared memory so it also requries tuning
# define TOP_K 128 
//Amount of neighbours checked per pixel. Possible values are 4 and 8. Doubles heap size if 8
# define CONNECTIVITY 4
//The minimimum total change in the number of pixels assigned per iteration. Used as a convergence criterion
# define CONVERGENCE 0.01
#define SAFE_CALL() {                                          \
    cudaDeviceSynchronize();                                            \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

// Used to turn debugging on an off. Useful for when you want to tune the heap size in shared-memory
#define DEBUG_LEVEL 0

#if DEBUG_LEVEL >= 1
#define BENCHMARK_START(name) \
    auto name##_start = std::chrono::high_resolution_clock::now();

#define BENCHMARK_END(name) \
    auto name##_end = std::chrono::high_resolution_clock::now(); \
    std::cout << #name << ": " \
              << std::chrono::duration_cast<std::chrono::microseconds>( \
                     name##_end - name##_start).count() \
              << " us\n";
#else
#define BENCHMARK_START(name)
#define BENCHMARK_END(name)
#endif


typedef uint32_t uint;
using namespace std;

//constants used to travers neighbourhoods
__constant__ int dx8[8];
__constant__ int dy8[8];
__constant__ int dn8[8];
__constant__ int swidth;
 
void FindSeeds(float* img, const int width, const int height, const int numk, double* kx, double* ky, int* outnumk, uint* clusters_dim_y, uint* clusters_dim_x)
{
    const int sz = width*height;
    double gridstep = sqrt((double)(sz)/(double)(numk)) + 0.5;
    // Search on either side of the grid step to get close to the requested
    // number of superpixels
    if(1)
    {
        int minerr = 999999;
        double minstep = gridstep-1.0; double maxstep = gridstep+1.0;
        for(double x = minstep; x <= maxstep; x += 0.1)
        {
            int err = abs( (int)(0.5 + width/x)*(int)(0.5 + height/x) - numk);
            if(err < minerr)
            {
                minerr = err; gridstep = x;
            }
        }
    }

    double halfstep = gridstep/2.0;


    const int MOVE[9] = {-1, -1*width, 1, width, 0, -1*width - 1, -1*width + 1,
                        width + 1, width - 1};

    const uint maxiter = 5;
    const double maxerr = 0.1;

    //A heuristic method of finding good seeds, if your neihgbours are not similar to you move in the most similar direction
        
    int n = 0;
    for(double y = halfstep; y <= height; y += gridstep)
    {
        int yval = (int)y;
        if(yval < height)
        {
            for(double x = halfstep; x <= width; x += gridstep)
            {
                int xval = (int)x;
                if(xval < width)
                {
                    double err = 1e32;
                    int direction = 4;
                    uint dbg = 0;

                    int idx = yval*width + xval;

                    while(err > maxerr && dbg < maxiter)
                    {
                        idx += MOVE[direction];    

                        double temp = 0;
                        double minval = 1e32;
                        for(uint i = 0; i < 9; i++) 
                        {
                            if(idx + MOVE[i] >= sz || idx + MOVE[i] < 0)
                                continue;
                            double val = img[idx] - img[idx + MOVE[i]] + 
                                         img[idx + sz] - img[idx + MOVE[i] + sz] +
                                         img[idx + 2*sz] - img[idx + MOVE[i] + 2*sz]; 
                            val = val/3;

                            if(val < minval)
                            {
                                minval = val;
                                direction = i;
                            }
                            temp += val*val;
                        }
                        err = temp;

                            dbg++;
                        }
                        kx[n] = (double)(idx % width);
                        ky[n] = (double)(idx / width);
                        n++;
                }
            }
        }
    }
    *outnumk = n;
    *clusters_dim_y = (width-halfstep)/gridstep + 1; 
    *clusters_dim_x = (height-halfstep)/gridstep + 1; 
}

void push (hHEAP *h, const int ind, const unsigned int klab, const double dist)
{
    if (h->len + 1 >= h->size)
    {
        printf("exceeded heap size\n");
        exit(1);
    }
    int i = h->len + 1;
    int j = i / 2;
    while (i > 1 && h->nodes[j].d > dist)
    {
        h->nodes[i] = h->nodes[j];
        i = j;
        j = j / 2;
    }
    h->nodes[i].i = ind;
    h->nodes[i].k = klab;
    h->nodes[i].d = dist;
    h->len++;
} 

//same as push just on CUDA
__device__ void dpush (hNODE* heap, uint heap_size, uint* heap_len, const int ind, const unsigned int klab, const double dist)
{
    uint templen = *heap_len;
    if(templen >= heap_size)
    {
        printf("exceeded heap size\n");
    }

    int i = templen + 1;
    int j = i / 2;
    while (i > 1 && heap[j].d > dist)
    {
        heap[i] = heap[j];
        i = j;
        j = j / 2;
    }
    heap[i].i = ind;
    heap[i].k = klab;
    heap[i].d = dist;
    templen++;

    *heap_len = templen;
}

//similar do dpush
__device__ void dpop(hNODE* heap, uint heap_size, uint* heap_len, int* ind, unsigned int* klab, double* dist)
{
    uint templen = *heap_len;
    if(templen > 1)
    {
        *ind = heap[1].i;
        *klab = heap[1].k;
        *dist = heap[1].d;

        int i, j, k;
        //int i = heap[1].i;
             
        heap[1] = heap[templen];
     
        templen--;
     
        i = 1;
        while (i!=templen+1)
        {
            k = templen+1;
            j = 2 * i;
            if (j <= templen && heap[j].d < heap[k].d)
            {
                k = j;
            }
            if (j + 1 <= templen && heap[j + 1].d < heap[k].d)
            {
                k = j + 1;
            }
            heap[i] = heap[k];
            i = k;
        }
    }
    else
    {
        *ind = -1;
    }

    *heap_len = templen;
}


/**
 * This kernel loads a block-specific priority queue (heap) from global to shared memory.
 * Thread 0 pops the highest priority candidate pixels, and threads concurrently attempt to
 * claim them via atomic operations. If a pixel is successfully claimed (previously unvisited),
 * the kernel dynamically updates the running spatial and color centroids for the assigned cluster.
 * Finally, the N-connected neighbors of the newly claimed pixel are evaluated, and valid
 * unassigned neighbors are pushed back into the heap based on their normalized SNIC distance.
 *
 * @param img          CUDA texture object bound to the input image (expected as float4).
 * @param nchans       Number of image color channels.
 * @param width        Image width in pixels.
 * @param height       Image height in pixels.
 * @param heap         Global memory array containing the priority queue nodes (hNODE) for all blocks.
 * @param heap_size    Maximum node capacity of the heap allocated per block.
 * @param heap_lens    Global array tracking the current active length of each block's heap.
 * @param toBePushed   Global buffer for nodes pending insertion.
 * @param labels       Global array storing the assigned cluster ID for each pixel (-1 if unassigned).
 * @param xs           Accumulated x-coordinates for each cluster centroid (double precision).
 * @param ys           Accumulated y-coordinates for each cluster centroid (double precision).
 * @param sizes        Current total number of pixels assigned to each cluster.
 * @param colors       Accumulated color channels for each cluster (flattened: numClusters * nchans).
 * @param pixelCount   Global atomic counter tracking the total number of successfully labeled pixels.
 * @param numClusters  Total number of initialized superpixel clusters.
 * @param invwt        Inverse spatial weight factor to balance color vs. spatial distance (compactness).
 */
__global__ void step(cudaTextureObject_t img,
                            const uint nchans, 
                            const uint width, 
                            const uint height,
                            hNODE* heap, 
                            const uint heap_size,
                            uint* heap_lens,
                            hNODE* toBePushed, 
                            int* labels, 
                            double* xs, 
                            double* ys,
                            uint* sizes, 
                            float* colors, 
                            unsigned int* pixelCount, 
                            const uint numClusters,
                            const int invwt)
{
    const uint tid = threadIdx.x;
    const uint heap_offset = blockIdx.x*heap_size;

    extern __shared__ hNODE b_heap[];
    __shared__ uint b_heap_len;
	__shared__ hNODE toBeProcessed[TOP_K];
    __shared__ hNODE toBePushedShared[TOP_K*CONNECTIVITY];

    //let thread 0 update current heap lengths
    if(tid == 0) b_heap_len = heap_lens[blockIdx.x];
    __syncthreads();

    //let threads assign the starting point of each blocks heap to its own block in parallel
    for(uint i = tid; i < b_heap_len + 1; i+=blockDim.x)
    {
        b_heap[i] = heap[i + heap_offset]; 
    }

    //init to be pushed
    for(int p = tid; p < blockDim.x*CONNECTIVITY; p += blockDim.x)
    {
        toBePushedShared[p].i = -1;
        toBePushedShared[p].d = 1e32;
    }

    if(tid == 0)
    {
        for(uint i = 0; i < blockDim.x; i++)
        {
            //pop elements using thread 0 for processing 
            dpop(b_heap, heap_size, &b_heap_len, &toBeProcessed[i].i, &toBeProcessed[i].k, &toBeProcessed[i].d);
        }
    }

    __syncthreads();

    hNODE local;
    local.i = toBeProcessed[tid].i;
    local.k = toBeProcessed[tid].k;
    local.d = toBeProcessed[tid].d;

    __syncthreads();

    //do the update to cluster centers in parallel using atomic operations to prevent tearing

    if(local.i >= 0)
    {
        const int k = local.k;
        const int x = local.i % width;
        const int y = local.i / width;
        const int i = y*width+x;


        if(atomicCAS(&labels[i], -1, (int)k) < 0)
        {
            atomicAdd(pixelCount, (unsigned int)1);

            float4 pixel = tex2D<float4>(img, (float)x + 0.5, (float)y + 0.5);

            const float c1 = atomicAdd(&colors[k + 0*numClusters], pixel.x) + 
                pixel.x;
            const float c2 = atomicAdd(&colors[k + 1*numClusters], pixel.y) + 
                pixel.y;
            const float c3 = atomicAdd(&colors[k + 2*numClusters], pixel.z) + 
                pixel.z;

            const double px = atomicAdd(&xs[k], (double)x) + x;
            const double py = atomicAdd(&ys[k], y) + (double)y;

            const float ksize = (float)atomicAdd(&sizes[k], 1) + 1.0f;


            int xx;
            int yy;
            int ii;

            //start looking for eligible neigbours

            for(int p = 0; p < CONNECTIVITY; p++)
            {
                xx = x + dx8[p];
                yy = y + dy8[p];
                ii = i + dn8[p];
                if(atomicAdd(&labels[ii], 0) < 0)
                {
                    float4 npixel = tex2D<float4>(img, (float)xx + 0.5, 
                            (float)yy + 0.5);
                    float dcx = c1 - npixel.x*ksize;
                    float dcy = c2 - npixel.y*ksize;
                    float dcz = c3 - npixel.z*ksize;

                    float colordist = (dcx*dcx + dcy*dcy + dcz*dcz);

                    int xdiff = px - xx*ksize;
                    int ydiff = py - yy*ksize;
                    uint xydist = xdiff*xdiff + ydiff*ydiff;

                    float slicdist = (colordist + xydist*invwt)
                        /(ksize*ksize);//late normalization by ksize[k], to have only one division operation

                    toBePushedShared[tid*CONNECTIVITY + p].i = ii;
                    toBePushedShared[tid*CONNECTIVITY + p].k = k;
                    toBePushedShared[tid*CONNECTIVITY + p].d = slicdist;
                }
            }
        }
    }   

    __syncthreads();

    if(tid == 0)
    {
        for(int i = 0; i < blockDim.x*CONNECTIVITY; i++)
        {
            if(toBePushedShared[i].i >= 0)
            {
                dpush(b_heap, heap_size, &b_heap_len, toBePushedShared[i].i, 
                                                toBePushedShared[i].k,
                                                toBePushedShared[i].d);
            }
        }

        heap_lens[blockIdx.x] = b_heap_len;
    }

    __syncthreads();

    for(uint i = tid; i < b_heap_len + 1; i+=blockDim.x)
    {
        heap[i + heap_offset].i = b_heap[i].i; 
        heap[i + heap_offset].k = b_heap[i].k;
        heap[i + heap_offset].d = b_heap[i].d;
    }
}

//I would be lying if I said I remember how texture objects work but essentialy they allow you to fetch multiple pixels at once which is great for cache efficiency
typedef struct texObjHandle_t {
    cudaTextureObject_t texObj;
    cudaArray_t cuArray;
} texObjHandle_t;

texObjHandle_t createTexObj(float* chans, const uint size, const uint width, 
        const uint height)
{
    const uint channels = 4;
    float* pixelPacked = new float[size * channels];

    for(uint i=0; i < size; i++)
    {
        pixelPacked[i * channels + 0] = chans[i + 0 * size];
        pixelPacked[i * channels + 1] = chans[i + 1 * size];
        pixelPacked[i * channels + 2] = chans[i + 2 * size];
        pixelPacked[i * channels + 3] = 1.0f;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, 
            cudaChannelFormatKindFloat);
    cudaArray_t cuArray;

    float4* inimg = (float4*)pixelPacked;

    cudaMallocArray(&cuArray, &channelDesc, channels*width, height);
    SAFE_CALL();

    const size_t spitch = channels * width * sizeof(float);

    cudaMemcpy2DToArray(cuArray, 0, 0, inimg, spitch, width * 4 *sizeof(float), 
            height, cudaMemcpyHostToDevice);
    SAFE_CALL();

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    SAFE_CALL();

    delete[] pixelPacked;
    texObjHandle_t handle;
    handle.texObj = texObj;
    handle.cuArray = cuArray;

    return handle;
}

/*
   float* chans: the flattened image
   int sz: size of the image in pixels
   uint width: width of the image in pixels
   uint height: height of the image in pixels
   const int innumk: number of desired superpixels
   int* outnumk: number of realized superpixels
   double compactnes: hyper-parameter describing the weighting of colorspace distance

*/
int* segment(float* chans, int sz, uint width, uint height,
        const int innumk, int* outnumk, double compactness)
{
    BENCHMARK_START(initialize)
    //maximum allowed shared memory of the device
    int maxoptin;
    cudaDeviceGetAttribute(&maxoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    //cut it down a little because it overflows if not due to the threads reserving a little bit of shared mem for themselves
    //feel free to change this fp value of 0.75 to according to your devices capabilities
	maxoptin = 0.75*(float)maxoptin;
	if(DEBUG_LEVEL == 1)
	{
    	cout << "optin " << maxoptin/1024 << endl;
	}
    //actualy reserves the shared mem
    cudaFuncSetAttribute(step, cudaFuncAttributeMaxDynamicSharedMemorySize, 
            maxoptin);
    SAFE_CALL();

	cudaFuncSetAttribute(step,
                     cudaFuncAttributePreferredSharedMemoryCarveout,
                     cudaSharedmemCarveoutMaxShared);
	SAFE_CALL();

    const uint nchans = 3;
    //get the size of the heaps in nodes
    const uint b_heap_size = maxoptin/sizeof(hNODE); 
	const int blocksraw = (int)(CONNECTIVITY*sz*sizeof(float))/(float)(b_heap_size*sizeof(hNODE) + 0.5);
    dim3 blockDims(sqrt(blocksraw), sqrt(blocksraw));
    const int blocks = blockDims.x*blockDims.y;

	if(DEBUG_LEVEL == 1)
	{
    cout << "heap size in kB " << b_heap_size*sizeof(hNODE)/1024 << endl;
	cout << "blocks x " << blockDims.x << " blocks y " << blockDims.y << endl; 
	}

    hHEAP** h_heaps = (hHEAP **)malloc(blocks*sizeof(hHEAP*)); 



    for(int i = 0; i < blocks; i++)
    {
        h_heaps[i] = (hHEAP *)malloc(sizeof(hHEAP));
        h_heaps[i]->len = 0;
        h_heaps[i]->size = b_heap_size;
        h_heaps[i]->nodes = (hNODE *)malloc(b_heap_size*sizeof(hNODE));
        h_heaps[i]->nodes[0].i = -1; //dummy node
    }

    int* h_labels = new int[sz];  
    double* h_xs = new double[(int)(innumk*1.1 + 10)];
    double* h_ys = new double[(int)(innumk*1.1 + 10)]; 
    uint* h_sizes = new uint[(int)(innumk*1.1 + 10)]; 
    float* h_colors = new float[(int)(innumk*1.1 + 10) * 3]; 

    int numk = 0;
    uint clusters_dim_y = 0;
    uint clusters_dim_x = 0;

    FindSeeds(chans, width, height, innumk, h_xs, h_ys, 
            &numk, &clusters_dim_y,&clusters_dim_x);


    uint* heap_idx = new uint[numk];

    uint cluster_blockdim_x = (uint)(width/ blockDims.x) + 1;
    uint cluster_blockdim_y = (uint)(height/ blockDims.y) + 1;

    //assign each block its heap
    for(uint i = 0; i < numk; i++)
    {
        uint pos = (int)(h_ys[i]*width + h_xs[i]);
        h_sizes[i] = 1;
        h_colors[i + 0*numk] = chans[pos + 0*sz];
        h_colors[i + 1*numk] = chans[pos + 1*sz];
        h_colors[i + 2*numk] = chans[pos + 2*sz];

        uint heap_x = (uint)(h_xs[i] / cluster_blockdim_x);
        uint heap_y = (uint)(h_ys[i] / cluster_blockdim_y);
        uint idx = min(heap_y*blockDims.x + heap_x, blocks - 1);

        push(h_heaps[idx], pos, i, 0);
        heap_idx[i] = heap_y*blockDims.x + heap_x;
    }
    
    hNODE* d_heap;
    const uint heap_size = b_heap_size;
    uint* d_heap_lens;
    hNODE* toBePushed;
    int* d_labels;
    double* d_xs;
    double* d_ys;
    uint* d_sizes;
    float* d_colors;
    unsigned int* d_pixelCount;


    //allocate gpu mem
    cudaMalloc((void**)&d_heap, heap_size*blocks*sizeof(hNODE));
    SAFE_CALL();
    cudaMalloc((void**)&d_heap_lens, blocks*sizeof(uint));
    SAFE_CALL();
    cudaMalloc((void**)&toBePushed, CONNECTIVITY*blocks*TOP_K*sizeof(hNODE));
    SAFE_CALL();
    cudaMalloc((void**)&d_labels, sz*sizeof(int));
    SAFE_CALL();
    cudaMalloc((void**)&d_xs, numk*sizeof(double));
    SAFE_CALL();
    cudaMalloc((void**)&d_ys, numk*sizeof(double));
    SAFE_CALL();
    cudaMalloc((void**)&d_sizes, numk*sizeof(uint));
    SAFE_CALL();
    cudaMalloc((void**)&d_colors, numk*nchans*sizeof(float));
    SAFE_CALL();
    cudaMalloc((void**)&d_pixelCount, sizeof(unsigned int));
    SAFE_CALL();

    hNODE* flat_heap = (hNODE*)malloc(heap_size*blocks*sizeof(hNODE));
    uint* flat_heap_lens = (uint*)malloc(blocks*sizeof(uint));

    //flatten the heaps as cuda doesnt allow 2d arrays
    
    for(int i = 0; i < blocks; i++)
    {
        for(uint j = 0; j < h_heaps[i]->size; j++)
        {
            flat_heap[i*heap_size + j].i = h_heaps[i]->nodes[j].i;
            flat_heap[i*heap_size + j].k = h_heaps[i]->nodes[j].k;
            flat_heap[i*heap_size + j].d = h_heaps[i]->nodes[j].d;
            if(j > h_heaps[i]->len + 1)
                flat_heap[i*heap_size + j].i = -1;
                flat_heap[i*heap_size + j].d = 1e32;

        }
        flat_heap[i*heap_size + 0].i = -1; //dummy node
        flat_heap_lens[i] = h_heaps[i]->len;
    }

    int dx8_host[8] = {-1,  0, 1, 0, -1,  1, 1, -1};//for 4 or 8 connectivity
    int dy8_host[8] = { 0, -1, 0, 1, -1, -1, 1,  1};//for 4 or 8 connectivity
    int swidth_host = (int)width;
    int dn8_host[8] = {-1, -swidth_host, 1, swidth_host, 
        -1-swidth_host, 1-swidth_host, 1+swidth_host, -1+swidth_host};

    cudaMemcpyToSymbol(dx8, dx8_host, 8*sizeof(int));
    SAFE_CALL();
    cudaMemcpyToSymbol(dy8, dy8_host, 8*sizeof(int));
    SAFE_CALL();
    cudaMemcpyToSymbol(dn8, dn8_host, 8*sizeof(int));
    SAFE_CALL();
    cudaMemcpyToSymbol(swidth, &swidth_host, sizeof(int));
    SAFE_CALL();

    //copy and set necessary parts
    texObjHandle_t handle = createTexObj(chans, sz, width, height);
    cudaTextureObject_t d_img = handle.texObj;
    cudaMemcpy(d_heap, flat_heap, heap_size*blocks*sizeof(hNODE),
            cudaMemcpyHostToDevice);
    SAFE_CALL();
    cudaMemcpy(d_heap_lens, flat_heap_lens, blocks*sizeof(uint),
            cudaMemcpyHostToDevice);
    SAFE_CALL();
    cudaMemset(d_labels, -1, sz*sizeof(int));
    SAFE_CALL();
    cudaMemcpy(d_xs, h_xs, numk*sizeof(double), cudaMemcpyHostToDevice);
    SAFE_CALL();
    cudaMemcpy(d_ys, h_ys, numk*sizeof(double), cudaMemcpyHostToDevice);
    SAFE_CALL();
    cudaMemcpy(d_sizes, h_sizes, numk*sizeof(uint), cudaMemcpyHostToDevice);
    SAFE_CALL();
    cudaMemcpy(d_colors, h_colors, numk*nchans*sizeof(float),
            cudaMemcpyHostToDevice);
    SAFE_CALL();
    cudaMemset(d_pixelCount, 0, sizeof(unsigned int));
    SAFE_CALL();

    const int invwt = compactness*compactness*numk/(double)sz; 

    uint oldGlobalPixelCount = 0;
    uint globalPixelCount = 0;
	uint iters = 0;

    BENCHMARK_END(initialize)

    BENCHMARK_START(step)
    auto bench_start = std::chrono::high_resolution_clock::now();
    step<<<blocks, TOP_K, heap_size*sizeof(hNODE)>>>(d_img, nchans, width,
            height, d_heap,heap_size, d_heap_lens,toBePushed, 
            d_labels, d_xs, d_ys, d_sizes, d_colors, d_pixelCount, numk, invwt);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "step kernel failed'%s'\n", 
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(&globalPixelCount, d_pixelCount, sizeof(uint),
            cudaMemcpyDeviceToHost);

    double convergence = (double)(globalPixelCount - oldGlobalPixelCount);

    while(convergence > CONVERGENCE)
    {
        step<<<blocks, TOP_K , heap_size*sizeof(hNODE)>>>(d_img, nchans, width,
            height, d_heap,heap_size, d_heap_lens, toBePushed, 
            d_labels, d_xs, d_ys, d_sizes, d_colors, d_pixelCount, numk, invwt);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "step kernel failed'%s'\n", 
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        oldGlobalPixelCount = globalPixelCount;
        cudaMemcpy(&globalPixelCount, d_pixelCount, sizeof(uint),
                cudaMemcpyDeviceToHost);
		iters++;
        convergence = (double)(globalPixelCount - oldGlobalPixelCount)/iters;
    }

    BENCHMARK_END(step)
    auto bench_end= std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(bench_end 
	- bench_start).count();

    /* leftover from benchmarking, can be ignored or deleted
	FILE* f = fopen("timingHD.log", "a");
	if(f)
	{
		fprintf(f, "%ld\n", elapsed);
		fclose(f);
	}
    */

    BENCHMARK_START(cleanup)
    //assign back to host mem
    cudaMemcpy(h_labels, d_labels, sz * sizeof(int), 
                cudaMemcpyDeviceToHost);
    SAFE_CALL();
    
    uint assnd = 0;
    for(int i = 0; i < sz; i++)
    {
        if(h_labels[i] >= 0)
            assnd++;
    }

    cudaFree(d_heap);
    SAFE_CALL();
    cudaFree(d_heap_lens);
    SAFE_CALL();
    cudaFree(toBePushed);
    SAFE_CALL();
    cudaFree(d_labels);
    SAFE_CALL();
    cudaFree(d_xs);
    SAFE_CALL();
    cudaFree(d_ys); 
    SAFE_CALL();
    cudaFree(d_sizes);
    SAFE_CALL();
    cudaFree(d_colors);
    SAFE_CALL();
    cudaFree(d_pixelCount);
    SAFE_CALL();
    cudaDestroyTextureObject(d_img);
    SAFE_CALL();
    cudaFreeArray(handle.cuArray);
    SAFE_CALL();
    free(flat_heap);
    free(flat_heap_lens);   
    delete[] h_xs;
    delete[] h_ys;
    delete[] h_sizes;
    delete[] h_colors;
    for(int i = 0; i < blocks; i++)
    {
        free(h_heaps[i]->nodes);
        free(h_heaps[i]);
    }
    free(h_heaps);
    BENCHMARK_END(cleanup)
    return h_labels;
}


void runSNICcuda(float* chans, const int nchans, const int width, 
                const int height, int* labels, int* outnumk, const int innumk,
                const double compactness)
{
    int* klabels = segment(chans, width*height, width, height, 
            innumk, outnumk, compactness); 

    //copy results to output
    if(klabels[0] < 0) klabels[0] = 0;
    for(int y = 1; y < height; y++)
    {
        for(int x = 1; x < width; x++)
        {
            int i = y*width+x;
            if(klabels[i] < 0)//find an adjacent label
            {
                if(klabels[i-1] >= 0) klabels[i] = klabels[i-1];
                else if(klabels[i-width] >= 0) klabels[i] = klabels[i-width];
            }//if labels[i] < 0 ends
        }
    }

    for(int i = 0; i < width*height; i++)
    {
        labels[i] = klabels[i];
    }

}
