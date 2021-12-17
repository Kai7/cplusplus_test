#include "imgwarp.hpp"

class WarpPerspectiveInvoker :
    public cv::ParallelLoopBody
{
public:
    WarpPerspectiveInvoker(const cv::Mat &_src, cv::Mat &_dst, const double *_M, int _interpolation,
                           int _borderType, const cv::Scalar &_borderValue) :
        ParallelLoopBody(), src(_src), dst(_dst), M(_M), interpolation(_interpolation),
        borderType(_borderType), borderValue(_borderValue)
    {
    }

    virtual void operator() (const cv::Range& range) const
    {
        const int BLOCK_SZ = 32;
        short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];
        int x, y, x1, y1, width = dst.cols, height = dst.rows;

        int bh0 = std::min(BLOCK_SZ/2, height);
        int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
        bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

        #if CV_SSE4_1
        // (OFF)
        #endif

        for( y = range.start; y < range.end; y += bh0 )
        {
            for( x = 0; x < width; x += bw0 )
            {
                int bw = std::min( bw0, width - x);
                int bh = std::min( bh0, range.end - y); // height

                cv::Mat _XY(bh, bw, CV_16SC2, XY), matA;
                cv::Mat dpart(dst, cv::Rect(x, y, bw, bh));

                for( y1 = 0; y1 < bh; y1++ )
                {
                    short* xy = XY + y1*bw*2;
                    double X0 = M[0]*x + M[1]*(y + y1) + M[2];
                    double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
                    double W0 = M[6]*x + M[7]*(y + y1) + M[8];

                    if( interpolation == INTER_NEAREST )
                    {
                        x1 = 0;

                        #if CV_SSE4_1
                        // (OFF)
                        #endif

                        for( ; x1 < bw; x1++ )
                        {
                            double W = W0 + M[6]*x1;
                            W = W ? 1./W : 0;
                            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
                            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
                            int X = cv::saturate_cast<int>(fX);
                            int Y = cv::saturate_cast<int>(fY);

                            xy[x1*2] = cv::saturate_cast<short>(X);
                            xy[x1*2+1] = cv::saturate_cast<short>(Y);
                        }
                    }
                    else
                    {
                        short* alpha = A + y1*bw;
                        x1 = 0;

                        #if CV_SSE4_1
                        // (OFF)
                        #endif

                        for( ; x1 < bw; x1++ )
                        {
                            double W = W0 + M[6]*x1;
                            W = W ? INTER_TAB_SIZE/W : 0;
                            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
                            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
                            int X = cv::saturate_cast<int>(fX);
                            int Y = cv::saturate_cast<int>(fY);

                            xy[x1*2] = cv::saturate_cast<short>(X >> INTER_BITS);
                            xy[x1*2+1] = cv::saturate_cast<short>(Y >> INTER_BITS);
                            alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
                                                (X & (INTER_TAB_SIZE-1)));
                        }
                    }
                }

                printf("[%s:%d]\n", __FILE__, __LINE__);
                if( interpolation == INTER_NEAREST ){
                    printf("case: INTER_NEAREST\n");
                    remap( src, dpart, _XY, Mat(), interpolation, borderType, borderValue );
                }
                else
                {
                    printf("case: not INTER_NEAREST\n");
                    cv::Mat _matA(bh, bw, CV_16U, A);
                    remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
                }
            }
        }
    }

private:
    cv::Mat src;
    cv::Mat dst;
    const double* M;
    int interpolation, borderType;
    cv::Scalar borderValue;
};

void warpPerspectve(int src_type,
                    const uchar * src_data, size_t src_step, int src_width, int src_height,
                    uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                    const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    /* [CV TRACE] Here! */
    printf("[%s:%d] hal::warpPerspective()\n", __FILE__, __LINE__);
    printf("src_type = %d\n", src_type);
    printf("borderType = %d\n", borderType);
    printf("borderValue[0] = %f\n", borderValue[0]);
    printf("borderValue[1] = %f\n", borderValue[1]);
    printf("borderValue[2] = %f\n", borderValue[2]);
    printf("borderValue[3] = %f\n", borderValue[3]);

    // CALL_HAL();

    cv::Mat src(cv::Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    cv::Mat dst(cv::Size(dst_width, dst_height), src_type, dst_data, dst_step);

    cv::Range range(0, dst.rows);
    WarpPerspectiveInvoker invoker(src, dst, M, interpolation, borderType, cv::Scalar(borderValue[0], borderValue[1], borderValue[2], borderValue[3]));
    cv::parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}


void warpPerspective( cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _M0,
                      cv::Size dsize, int flags, int borderType, const cv::Scalar& borderValue ) {

    printf("\n==================\n");
    printf("CV_8UC1 = %d\n", CV_8UC1);
    printf("CV_8UC2 = %d\n", CV_8UC2);
    printf("CV_8UC3 = %d\n", CV_8UC3);
    printf("CV_8SC1 = %d\n", CV_8SC1);
    printf("CV_8SC2 = %d\n", CV_8SC2);
    printf("CV_8SC3 = %d\n", CV_8SC3);
    printf("CV_16UC1 = %d\n", CV_16UC1);
    printf("CV_16UC2 = %d\n", CV_16UC2);
    printf("CV_16UC3 = %d\n", CV_16UC3);
    printf("CV_16SC1 = %d\n", CV_16SC1);
    printf("CV_16SC2 = %d\n", CV_16SC2);
    printf("CV_16SC3 = %d\n", CV_16SC3);
    printf("CV_32FC1 = %d\n", CV_32FC1);
    printf("CV_32FC2 = %d\n", CV_32FC2);
    printf("CV_32FC3 = %d\n", CV_32FC3);
    printf("==================\n\n");

    printf("[%s:%d] hal::cv::warpPerspective()\n", __FILE__, __LINE__);

#if CV_NEON
    // [CV TRACE] Here!
#endif

    assert( _src.total() > 0 );

    // CV_OCL_RUN()

    // CV_OCL_RUN()

    cv::Mat src = _src.getMat(), M0 = _M0.getMat();
    _dst.create( dsize.area() == 0 ? src.size() : dsize, src.type() );
    cv::Mat dst = _dst.getMat();

    if( dst.data == src.data )
        src = src.clone();

    double M[9];
    cv::Mat matM(3, 3, CV_64F, M);
    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;
    printf("interpolation = %d\n", interpolation);
    printf("borderType = %d\n", borderType);
    printf("borderValue[0] = %f\n", borderValue.val[0]);
    printf("borderValue[1] = %f\n", borderValue.val[1]);
    printf("borderValue[2] = %f\n", borderValue.val[2]);
    printf("borderValue[3] = %f\n", borderValue.val[3]);

    CV_Assert( (M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3 );
    M0.convertTo(matM, matM.type());

#if defined (HAVE_VIPP) && IPP_VERSION_X100 >= 810 && IPP_DISABLE_BLOCK
    // (OFF)
#endif

    if( !(flags & WARP_INVERSE_MAP) ){
        // [CV TEST] Here!
        invert(matM, matM);
    }

    warpPerspectve(src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows,
                        matM.ptr<double>(), interpolation, borderType, borderValue.val);
}

void ai::getPerspectiveTransform(){}
void ai::getAffineTransform(){}
void ai::warpPerspective(){}
void ai::warpAffine(){}