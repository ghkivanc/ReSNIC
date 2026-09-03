typedef struct
{
    int i; // the x and y values packed into one
    unsigned int k; // the label
    double d;       // the distance
} hNODE;

typedef struct
{
    hNODE *nodes;
    uint len; // number of elements present
    uint size; // total capacity in terms of memory allocated
} hHEAP;

void runSNICcuda(float* chans, const int nchans, const int width, 
                const int height, int* labels, int* outnumk, const int innumk,
                const double compactness);


int* testTwoPartKernel(float* chans, int size, uint width, uint height, const int innumk, int* outnumk, double compactness);
