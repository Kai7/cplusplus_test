#include "imgwarp.hpp"
// #include <algorithm>
#include "arm_neon.h"

#define INTERPOLATION_ADVANCED 0

/************** interpolation formulas and tables ***************/

const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

const int INTER_REMAP_COEF_BITS=15;
const int INTER_REMAP_COEF_SCALE=1 << INTER_REMAP_COEF_BITS;

static uchar NNDeltaTab_i[INTER_TAB_SIZE2][2];

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

static inline void interpolateLinear( float x, float* coeffs )
{
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static inline void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

#if 1
static short BilinearTab_iC4_buf[INTER_TAB_SIZE2+2][2][8];
static short (*BilinearTab_iC4)[2][8] = (short (*)[2][8])cv::alignPtr(BilinearTab_iC4_buf, 16);
#endif

#if INTERPOLATION_ADVANCED
static float BicubicTab_f[INTER_TAB_SIZE2][4][4];
static short BicubicTab_i[INTER_TAB_SIZE2][4][4];

static float Lanczos4Tab_f[INTER_TAB_SIZE2][8][8];
static short Lanczos4Tab_i[INTER_TAB_SIZE2][8][8];
#endif


static inline void interpolateLanczos4( float x, float* coeffs )
{
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2]=
    {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

    if( x < FLT_EPSILON )
    {
        for( int i = 0; i < 8; i++ )
            coeffs[i] = 0;
        coeffs[3] = 1;
        return;
    }

    float sum = 0;
    double y0=-(x+3)*CV_PI*0.25, s0 = sin(y0), c0=cos(y0);
    for(int i = 0; i < 8; i++ )
    {
        double y = -(x+3-i)*CV_PI*0.25;
        coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
        sum += coeffs[i];
    }

    sum = 1.f/sum;
    for(int i = 0; i < 8; i++ )
        coeffs[i] *= sum;
}

static void initInterTab1D(int method, float* tab, int tabsz)
{
    float scale = 1.f/tabsz;
    if( method == INTER_LINEAR )
    {
        for( int i = 0; i < tabsz; i++, tab += 2 )
            interpolateLinear( i*scale, tab );
    }
    else if( method == INTER_CUBIC )
    {
        for( int i = 0; i < tabsz; i++, tab += 4 )
            interpolateCubic( i*scale, tab );
    }
    else if( method == INTER_LANCZOS4 )
    {
        for( int i = 0; i < tabsz; i++, tab += 8 )
            interpolateLanczos4( i*scale, tab );
    }
    else
        CV_Error( CV_StsBadArg, "Unknown interpolation method" );
}

static const void* initInterTab2D( int method, bool fixpt )
{
    static bool inittab[INTER_MAX+1] = {false};
    float* tab = 0;
    short* itab = 0;
    int ksize = 0;
    if( method == INTER_LINEAR ){
        tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize=2;
    }
    else if( method == INTER_CUBIC ){
        assert(0);
        // tab = BicubicTab_f[0][0], itab = BicubicTab_i[0][0], ksize=4;
    }
    else if( method == INTER_LANCZOS4 ){
        assert(0);
        // tab = Lanczos4Tab_f[0][0], itab = Lanczos4Tab_i[0][0], ksize=8;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported interpolation type" );

    if( !inittab[method] )
    {
        cv::AutoBuffer<float> _tab(8*INTER_TAB_SIZE);
        int i, j, k1, k2;
        initInterTab1D(method, _tab, INTER_TAB_SIZE);
        for( i = 0; i < INTER_TAB_SIZE; i++ )
            for( j = 0; j < INTER_TAB_SIZE; j++, tab += ksize*ksize, itab += ksize*ksize )
            {
                int isum = 0;
                NNDeltaTab_i[i*INTER_TAB_SIZE+j][0] = j < INTER_TAB_SIZE/2;
                NNDeltaTab_i[i*INTER_TAB_SIZE+j][1] = i < INTER_TAB_SIZE/2;

                for( k1 = 0; k1 < ksize; k1++ )
                {
                    float vy = _tab[i*ksize + k1];
                    for( k2 = 0; k2 < ksize; k2++ )
                    {
                        float v = vy*_tab[j*ksize + k2];
                        tab[k1*ksize + k2] = v;
                        isum += itab[k1*ksize + k2] = cv::saturate_cast<short>(v*INTER_REMAP_COEF_SCALE);
                    }
                }

                if( isum != INTER_REMAP_COEF_SCALE )
                {
                    int diff = isum - INTER_REMAP_COEF_SCALE;
                    int ksize2 = ksize/2, Mk1=ksize2, Mk2=ksize2, mk1=ksize2, mk2=ksize2;
                    for( k1 = ksize2; k1 < ksize2+2; k1++ )
                        for( k2 = ksize2; k2 < ksize2+2; k2++ )
                        {
                            if( itab[k1*ksize+k2] < itab[mk1*ksize+mk2] )
                                mk1 = k1, mk2 = k2;
                            else if( itab[k1*ksize+k2] > itab[Mk1*ksize+Mk2] )
                                Mk1 = k1, Mk2 = k2;
                        }
                    if( diff < 0 )
                        itab[Mk1*ksize + Mk2] = (short)(itab[Mk1*ksize + Mk2] - diff);
                    else
                        itab[mk1*ksize + mk2] = (short)(itab[mk1*ksize + mk2] - diff);
                }
            }
        tab -= INTER_TAB_SIZE2*ksize*ksize;
        itab -= INTER_TAB_SIZE2*ksize*ksize;
#if 0   /* CV_NEON */
        if( method == INTER_LINEAR )
        {
            for( i = 0; i < INTER_TAB_SIZE2; i++ )
                for( j = 0; j < 4; j++ )
                {
                    BilinearTab_iC4[i][0][j*2] = BilinearTab_i[i][0][0];
                    BilinearTab_iC4[i][0][j*2+1] = BilinearTab_i[i][0][1];
                    BilinearTab_iC4[i][1][j*2] = BilinearTab_i[i][1][0];
                    BilinearTab_iC4[i][1][j*2+1] = BilinearTab_i[i][1][1];
                }
        }
#endif
        inittab[method] = true;
    }
    return fixpt ? (const void*)itab : (const void*)tab;
}


template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return cv::saturate_cast<DT>(val); }
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const { return cv::saturate_cast<DT>((val + DELTA)>>SHIFT); }
};


static inline int clip(int x, int a, int b) {
  return x >= a ? (x < b ? x : b - 1) : a;
}


/****************************************************************************************\
*                       General warping (affine, perspective, remap)                     *
\****************************************************************************************/

template <typename T>
static void remapNearest(const cv::Mat &_src, cv::Mat &_dst, const cv::Mat &_xy,
                         int borderType, const cv::Scalar &_borderValue) {
  cv::Size ssize = _src.size(), dsize = _dst.size();
  int cn = _src.channels();
  const T *S0 = _src.ptr<T>();
  size_t sstep = _src.step / sizeof(S0[0]);
  cv::Scalar_<T> cval(
      cv::saturate_cast<T>(_borderValue[0]), cv::saturate_cast<T>(_borderValue[1]),
      cv::saturate_cast<T>(_borderValue[2]), cv::saturate_cast<T>(_borderValue[3]));
  int dx, dy;

  unsigned width1 = ssize.width, height1 = ssize.height;

  if (_dst.isContinuous() && _xy.isContinuous()) {
    dsize.width *= dsize.height;
    dsize.height = 1;
  }

  for (dy = 0; dy < dsize.height; dy++) {
    T *D = _dst.ptr<T>(dy);
    const short *XY = _xy.ptr<short>(dy);

    if (cn == 1) {
      for (dx = 0; dx < dsize.width; dx++) {
        int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
        if ((unsigned)sx < width1 && (unsigned)sy < height1)
          D[dx] = S0[sy * sstep + sx];
        else {
          if (borderType == cv::BORDER_REPLICATE) {
            sx = clip(sx, 0, ssize.width);
            sy = clip(sy, 0, ssize.height);
            D[dx] = S0[sy * sstep + sx];
          } else if (borderType == cv::BORDER_CONSTANT)
            D[dx] = cval[0];
          else if (borderType != cv::BORDER_TRANSPARENT) {
            sx = cv::borderInterpolate(sx, ssize.width, borderType);
            sy = cv::borderInterpolate(sy, ssize.height, borderType);
            D[dx] = S0[sy * sstep + sx];
          }
        }
      }
    } else {
      for (dx = 0; dx < dsize.width; dx++, D += cn) {
        int sx = XY[dx * 2], sy = XY[dx * 2 + 1], k;
        const T *S;
        if ((unsigned)sx < width1 && (unsigned)sy < height1) {
          if (cn == 3) {
            S = S0 + sy * sstep + sx * 3;
            D[0] = S[0], D[1] = S[1], D[2] = S[2];
          } else if (cn == 4) {
            S = S0 + sy * sstep + sx * 4;
            D[0] = S[0], D[1] = S[1], D[2] = S[2], D[3] = S[3];
          } else {
            S = S0 + sy * sstep + sx * cn;
            for (k = 0; k < cn; k++)
              D[k] = S[k];
          }
        } else if (borderType != cv::BORDER_TRANSPARENT) {
          if (borderType == cv::BORDER_REPLICATE) {
            sx = clip(sx, 0, ssize.width);
            sy = clip(sy, 0, ssize.height);
            S = S0 + sy * sstep + sx * cn;
          } else if (borderType == cv::BORDER_CONSTANT)
            S = &cval[0];
          else {
            sx = cv::borderInterpolate(sx, ssize.width, borderType);
            sy = cv::borderInterpolate(sy, ssize.height, borderType);
            S = S0 + sy * sstep + sx * cn;
          }
          for (k = 0; k < cn; k++)
            D[k] = S[k];
        }
      }
    }
  }
}
struct RemapNoVec {
    int operator()( const cv::Mat&, void*, const short*, const ushort*,
                    const void*, int ) const { return 0; }
};

typedef RemapNoVec RemapVec_8u;

template<class CastOp, class VecOp, typename AT>
static void remapBilinear( const cv::Mat& _src, cv::Mat& _dst, const cv::Mat& _xy,
                           const cv::Mat& _fxy, const void* _wtab,
                           int borderType, const cv::Scalar& _borderValue )
{
    typedef typename CastOp::rtype T;
    typedef typename CastOp::type1 WT;
    cv::Size ssize = _src.size(), dsize = _dst.size();
    int k, cn = _src.channels();
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = _src.ptr<T>();
    size_t sstep = _src.step/sizeof(S0[0]);
    T cval[CV_CN_MAX];
    int dx, dy;
    CastOp castOp;
    VecOp vecOp;

    for( k = 0; k < cn; k++ )
        cval[k] = cv::saturate_cast<T>(_borderValue[k & 3]);

    unsigned width1 = std::max(ssize.width-1, 0), height1 = std::max(ssize.height-1, 0);
    CV_Assert( ssize.area() > 0 );

#if CV_SSE2
// (OFF)
#endif

    for( dy = 0; dy < dsize.height; dy++ )
    {
        T* D = _dst.ptr<T>(dy);
        const short* XY = _xy.ptr<short>(dy);
        const ushort* FXY = _fxy.ptr<ushort>(dy);
        int X0 = 0;
        bool prevInlier = false;

        for( dx = 0; dx <= dsize.width; dx++ )
        {
            bool curInlier = dx < dsize.width ?
                (unsigned)XY[dx*2] < width1 &&
                (unsigned)XY[dx*2+1] < height1 : !prevInlier;
            if( curInlier == prevInlier )
                continue;

            int X1 = dx;
            dx = X0;
            X0 = X1;
            prevInlier = curInlier;

            if( !curInlier )
            {
                int len = vecOp( _src, D, XY + dx*2, FXY + dx, wtab, X1 - dx );
                D += len*cn;
                dx += len;

                if( cn == 1 )
                {
                    for( ; dx < X1; dx++, D++ )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx;
                        *D = castOp(WT(S[0]*w[0] + S[1]*w[1] + S[sstep]*w[2] + S[sstep+1]*w[3]));
                    }
                }
                else if( cn == 2 )
                    for( ; dx < X1; dx++, D += 2 )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*2;
                        WT t0 = S[0]*w[0] + S[2]*w[1] + S[sstep]*w[2] + S[sstep+2]*w[3];
                        WT t1 = S[1]*w[0] + S[3]*w[1] + S[sstep+1]*w[2] + S[sstep+3]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1);
                    }
                else if( cn == 3 )
                    for( ; dx < X1; dx++, D += 3 )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*3;
                        WT t0 = S[0]*w[0] + S[3]*w[1] + S[sstep]*w[2] + S[sstep+3]*w[3];
                        WT t1 = S[1]*w[0] + S[4]*w[1] + S[sstep+1]*w[2] + S[sstep+4]*w[3];
                        WT t2 = S[2]*w[0] + S[5]*w[1] + S[sstep+2]*w[2] + S[sstep+5]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1); D[2] = castOp(t2);
                    }
                else if( cn == 4 )
                    for( ; dx < X1; dx++, D += 4 )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*4;
                        WT t0 = S[0]*w[0] + S[4]*w[1] + S[sstep]*w[2] + S[sstep+4]*w[3];
                        WT t1 = S[1]*w[0] + S[5]*w[1] + S[sstep+1]*w[2] + S[sstep+5]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1);
                        t0 = S[2]*w[0] + S[6]*w[1] + S[sstep+2]*w[2] + S[sstep+6]*w[3];
                        t1 = S[3]*w[0] + S[7]*w[1] + S[sstep+3]*w[2] + S[sstep+7]*w[3];
                        D[2] = castOp(t0); D[3] = castOp(t1);
                    }
                else
                    for( ; dx < X1; dx++, D += cn )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*cn;
                        for( k = 0; k < cn; k++ )
                        {
                            WT t0 = S[k]*w[0] + S[k+cn]*w[1] + S[sstep+k]*w[2] + S[sstep+k+cn]*w[3];
                            D[k] = castOp(t0);
                        }
                    }
            }
            else
            {
                if( borderType == cv::BORDER_TRANSPARENT && cn != 3 )
                {
                    D += (X1 - dx)*cn;
                    dx = X1;
                    continue;
                }

                if( cn == 1 )
                    for( ; dx < X1; dx++, D++ )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        if( borderType == cv::BORDER_CONSTANT &&
                            (sx >= ssize.width || sx+1 < 0 ||
                             sy >= ssize.height || sy+1 < 0) )
                        {
                            D[0] = cval[0];
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            T v0, v1, v2, v3;
                            const AT* w = wtab + FXY[dx]*4;
                            if( borderType == cv::BORDER_REPLICATE )
                            {
                                sx0 = clip(sx, 0, ssize.width);
                                sx1 = clip(sx+1, 0, ssize.width);
                                sy0 = clip(sy, 0, ssize.height);
                                sy1 = clip(sy+1, 0, ssize.height);
                                v0 = S0[sy0*sstep + sx0];
                                v1 = S0[sy0*sstep + sx1];
                                v2 = S0[sy1*sstep + sx0];
                                v3 = S0[sy1*sstep + sx1];
                            }
                            else
                            {
                                sx0 = cv::borderInterpolate(sx, ssize.width, borderType);
                                sx1 = cv::borderInterpolate(sx+1, ssize.width, borderType);
                                sy0 = cv::borderInterpolate(sy, ssize.height, borderType);
                                sy1 = cv::borderInterpolate(sy+1, ssize.height, borderType);
                                v0 = sx0 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx0] : cval[0];
                                v1 = sx1 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx1] : cval[0];
                                v2 = sx0 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx0] : cval[0];
                                v3 = sx1 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx1] : cval[0];
                            }
                            D[0] = castOp(WT(v0*w[0] + v1*w[1] + v2*w[2] + v3*w[3]));
                        }
                    }
                else
                    for( ; dx < X1; dx++, D += cn )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        if( borderType == cv::BORDER_CONSTANT &&
                            (sx >= ssize.width || sx+1 < 0 ||
                             sy >= ssize.height || sy+1 < 0) )
                        {
                            for( k = 0; k < cn; k++ )
                                D[k] = cval[k];
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            const T *v0, *v1, *v2, *v3;
                            const AT* w = wtab + FXY[dx]*4;
                            if( borderType == cv::BORDER_REPLICATE )
                            {
                                sx0 = clip(sx, 0, ssize.width);
                                sx1 = clip(sx+1, 0, ssize.width);
                                sy0 = clip(sy, 0, ssize.height);
                                sy1 = clip(sy+1, 0, ssize.height);
                                v0 = S0 + sy0*sstep + sx0*cn;
                                v1 = S0 + sy0*sstep + sx1*cn;
                                v2 = S0 + sy1*sstep + sx0*cn;
                                v3 = S0 + sy1*sstep + sx1*cn;
                            }
                            else if( borderType == cv::BORDER_TRANSPARENT &&
                                ((unsigned)sx >= (unsigned)(ssize.width-1) ||
                                (unsigned)sy >= (unsigned)(ssize.height-1)))
                                continue;
                            else
                            {
                                sx0 = cv::borderInterpolate(sx, ssize.width, borderType);
                                sx1 = cv::borderInterpolate(sx+1, ssize.width, borderType);
                                sy0 = cv::borderInterpolate(sy, ssize.height, borderType);
                                sy1 = cv::borderInterpolate(sy+1, ssize.height, borderType);
                                v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx0*cn : &cval[0];
                                v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx1*cn : &cval[0];
                                v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx0*cn : &cval[0];
                                v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx1*cn : &cval[0];
                            }
                            for( k = 0; k < cn; k++ )
                                D[k] = castOp(WT(v0[k]*w[0] + v1[k]*w[1] + v2[k]*w[2] + v3[k]*w[3]));
                        }
                    }
            }
        }
    }
}

typedef void (*RemapNNFunc)(const cv::Mat& _src, cv::Mat& _dst, const cv::Mat& _xy,
                            int borderType, const cv::Scalar& _borderValue );

typedef void (*RemapFunc)(const cv::Mat& _src, cv::Mat& _dst, const cv::Mat& _xy,
                          const cv::Mat& _fxy, const void* _wtab,
                          int borderType, const cv::Scalar& _borderValue);

class RemapInvoker :
    public cv::ParallelLoopBody
{
public:
    RemapInvoker(const cv::Mat& _src, cv::Mat& _dst, const cv::Mat *_m1,
                 const cv::Mat *_m2, int _borderType, const cv::Scalar &_borderValue,
                 int _planar_input, RemapNNFunc _nnfunc, RemapFunc _ifunc, const void *_ctab) :
        ParallelLoopBody(), src(&_src), dst(&_dst), m1(_m1), m2(_m2),
        borderType(_borderType), borderValue(_borderValue),
        planar_input(_planar_input), nnfunc(_nnfunc), ifunc(_ifunc), ctab(_ctab)
    {
    }

    virtual void operator() (const cv::Range& range) const
    {
        int x, y, x1, y1;
        const int buf_size = 1 << 14;
        int brows0 = std::min(128, dst->rows), map_depth = m1->depth();
        int bcols0 = std::min(buf_size/brows0, dst->cols);
        brows0 = std::min(buf_size/bcols0, dst->rows);
    #if CV_SSE2
    #endif

        cv::Mat _bufxy(brows0, bcols0, CV_16SC2), _bufa;
        if( !nnfunc )
            _bufa.create(brows0, bcols0, CV_16UC1);

        for( y = range.start; y < range.end; y += brows0 )
        {
            for( x = 0; x < dst->cols; x += bcols0 )
            {
                int brows = std::min(brows0, range.end - y);
                int bcols = std::min(bcols0, dst->cols - x);
                cv::Mat dpart(*dst, cv::Rect(x, y, bcols, brows));
                cv::Mat bufxy(_bufxy, cv::Rect(0, 0, bcols, brows));

                if( nnfunc )
                {
                    if( m1->type() == CV_16SC2 && m2->empty() ) // the data is already in the right format
                        bufxy = (*m1)(cv::Rect(x, y, bcols, brows));
                    else if( map_depth != CV_32F )
                    {
                        for( y1 = 0; y1 < brows; y1++ )
                        {
                            short* XY = bufxy.ptr<short>(y1);
                            const short* sXY = m1->ptr<short>(y+y1) + x*2;
                            const ushort* sA = m2->ptr<ushort>(y+y1) + x;

                            for( x1 = 0; x1 < bcols; x1++ )
                            {
                                int a = sA[x1] & (INTER_TAB_SIZE2-1);
                                XY[x1*2] = sXY[x1*2] + NNDeltaTab_i[a][0];
                                XY[x1*2+1] = sXY[x1*2+1] + NNDeltaTab_i[a][1];
                            }
                        }
                    }
                    else if( !planar_input )
                        (*m1)(cv::Rect(x, y, bcols, brows)).convertTo(bufxy, bufxy.depth());
                    else
                    {
                        for( y1 = 0; y1 < brows; y1++ )
                        {
                            short* XY = bufxy.ptr<short>(y1);
                            const float* sX = m1->ptr<float>(y+y1) + x;
                            const float* sY = m2->ptr<float>(y+y1) + x;
                            x1 = 0;

                        #if CV_SSE2
                        #endif

                            for( ; x1 < bcols; x1++ )
                            {
                                XY[x1*2] = cv::saturate_cast<short>(sX[x1]);
                                XY[x1*2+1] = cv::saturate_cast<short>(sY[x1]);
                            }
                        }
                    }
                    nnfunc( *src, dpart, bufxy, borderType, borderValue );
                    continue;
                }

                cv::Mat bufa(_bufa, cv::Rect(0, 0, bcols, brows));
                for( y1 = 0; y1 < brows; y1++ )
                {
                    short* XY = bufxy.ptr<short>(y1);
                    ushort* A = bufa.ptr<ushort>(y1);

                    if( m1->type() == CV_16SC2 && (m2->type() == CV_16UC1 || m2->type() == CV_16SC1) )
                    {
                        bufxy = (*m1)(cv::Rect(x, y, bcols, brows));

                        const ushort* sA = m2->ptr<ushort>(y+y1) + x;
                        x1 = 0;

                    #if 0   /* CV_NEON */
                        uint16x8_t v_scale = vdupq_n_u16(INTER_TAB_SIZE2-1);
                        for ( ; x1 <= bcols - 8; x1 += 8)
                            vst1q_u16(A + x1, vandq_u16(vld1q_u16(sA + x1), v_scale));
                    #endif

                        for( ; x1 < bcols; x1++ )
                            A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2-1));
                    }
                    else if( planar_input )
                    {
                        const float* sX = m1->ptr<float>(y+y1) + x;
                        const float* sY = m2->ptr<float>(y+y1) + x;

                        x1 = 0;

                    #if 0   /* CV_NEON */
                        float32x4_t v_scale = vdupq_n_f32((float)INTER_TAB_SIZE);
                        int32x4_t v_scale2 = vdupq_n_s32(INTER_TAB_SIZE - 1), v_scale3 = vdupq_n_s32(INTER_TAB_SIZE);

                        for( ; x1 <= bcols - 4; x1 += 4 )
                        {
                            int32x4_t v_sx = cv_vrndq_s32_f32(vmulq_f32(vld1q_f32(sX + x1), v_scale)),
                                      v_sy = cv_vrndq_s32_f32(vmulq_f32(vld1q_f32(sY + x1), v_scale));
                            int32x4_t v_v = vmlaq_s32(vandq_s32(v_sx, v_scale2), v_scale3,
                                                      vandq_s32(v_sy, v_scale2));
                            vst1_u16(A + x1, vqmovun_s32(v_v));

                            int16x4x2_t v_dst = vzip_s16(vqmovn_s32(vshrq_n_s32(v_sx, INTER_BITS)),
                                                         vqmovn_s32(vshrq_n_s32(v_sy, INTER_BITS)));
                            vst1q_s16(XY + (x1 << 1), vcombine_s16(v_dst.val[0], v_dst.val[1]));
                        }
                    #endif

                        for( ; x1 < bcols; x1++ )
                        {
                            int sx = cvRound(sX[x1]*INTER_TAB_SIZE);
                            int sy = cvRound(sY[x1]*INTER_TAB_SIZE);
                            int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
                            XY[x1*2] = cv::saturate_cast<short>(sx >> INTER_BITS);
                            XY[x1*2+1] = cv::saturate_cast<short>(sy >> INTER_BITS);
                            A[x1] = (ushort)v;
                        }
                    }
                    else
                    {
                        const float* sXY = m1->ptr<float>(y+y1) + x*2;
                        x1 = 0;

                    #if 0   /* CV_NEON */
                        float32x4_t v_scale = vdupq_n_f32(INTER_TAB_SIZE);
                        int32x4_t v_scale2 = vdupq_n_s32(INTER_TAB_SIZE-1), v_scale3 = vdupq_n_s32(INTER_TAB_SIZE);

                        for( ; x1 <= bcols - 4; x1 += 4 )
                        {
                            float32x4x2_t v_src = vld2q_f32(sXY + (x1 << 1));
                            int32x4_t v_sx = cv_vrndq_s32_f32(vmulq_f32(v_src.val[0], v_scale));
                            int32x4_t v_sy = cv_vrndq_s32_f32(vmulq_f32(v_src.val[1], v_scale));
                            int32x4_t v_v = vmlaq_s32(vandq_s32(v_sx, v_scale2), v_scale3,
                                                      vandq_s32(v_sy, v_scale2));
                            vst1_u16(A + x1, vqmovun_s32(v_v));

                            int16x4x2_t v_dst = vzip_s16(vqmovn_s32(vshrq_n_s32(v_sx, INTER_BITS)),
                                                         vqmovn_s32(vshrq_n_s32(v_sy, INTER_BITS)));
                            vst1q_s16(XY + (x1 << 1), vcombine_s16(v_dst.val[0], v_dst.val[1]));
                        }
                    #endif

                        for( x1 = 0; x1 < bcols; x1++ )
                        {
                            int sx = cvRound(sXY[x1*2]*INTER_TAB_SIZE);
                            int sy = cvRound(sXY[x1*2+1]*INTER_TAB_SIZE);
                            int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
                            XY[x1*2] = cv::saturate_cast<short>(sx >> INTER_BITS);
                            XY[x1*2+1] = cv::saturate_cast<short>(sy >> INTER_BITS);
                            A[x1] = (ushort)v;
                        }
                    }
                }
                ifunc(*src, dpart, bufxy, bufa, ctab, borderType, borderValue);
            }
        }
    }

private:
    const cv::Mat* src;
    cv::Mat* dst;
    const cv::Mat *m1, *m2;
    int borderType;
    cv::Scalar borderValue;
    int planar_input;
    RemapNNFunc nnfunc;
    RemapFunc ifunc;
    const void *ctab;
};


void remap(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _map1,
           cv::InputArray _map2, int interpolation, int borderType,
           const cv::Scalar &borderValue) {
  static RemapNNFunc nn_tab[] = {remapNearest<uchar>,  remapNearest<schar>,
                                 remapNearest<ushort>, remapNearest<short>,
                                 remapNearest<int>,    remapNearest<float>,
                                 remapNearest<double>, 0};

  static RemapFunc linear_tab[] = {
      remapBilinear<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, RemapVec_8u,
                    short>,
      0,
      remapBilinear<Cast<float, ushort>, RemapNoVec, float>,
      remapBilinear<Cast<float, short>, RemapNoVec, float>,
      0,
      remapBilinear<Cast<float, float>, RemapNoVec, float>,
      remapBilinear<Cast<double, double>, RemapNoVec, float>,
      0};

#if 0
  static RemapFunc cubic_tab[] = {
      remapBicubic<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short,
                   INTER_REMAP_COEF_SCALE>,
      0,
      remapBicubic<Cast<float, ushort>, float, 1>,
      remapBicubic<Cast<float, short>, float, 1>,
      0,
      remapBicubic<Cast<float, float>, float, 1>,
      remapBicubic<Cast<double, double>, float, 1>,
      0};

  static RemapFunc lanczos4_tab[] = {
      remapLanczos4<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short,
                    INTER_REMAP_COEF_SCALE>,
      0,
      remapLanczos4<Cast<float, ushort>, float, 1>,
      remapLanczos4<Cast<float, short>, float, 1>,
      0,
      remapLanczos4<Cast<float, float>, float, 1>,
      remapLanczos4<Cast<double, double>, float, 1>,
      0};
#endif

  CV_Assert(_map1.size().area() > 0);
  CV_Assert(_map2.empty() || (_map2.size() == _map1.size()));

//   CV_OCL_RUN();

  cv::Mat src = _src.getMat(), map1 = _map1.getMat(), map2 = _map2.getMat();
  _dst.create(map1.size(), src.type());
  cv::Mat dst = _dst.getMat();

//   CV_OVX_RUN();

  CV_Assert(dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX &&
            src.rows < SHRT_MAX);

  if (dst.data == src.data)
    src = src.clone();

  if (interpolation == INTER_AREA)
    interpolation = INTER_LINEAR;

  int type = src.type(), depth = CV_MAT_DEPTH(type);
  printf("[%s:%d] type = %d\n", __FILE__, __LINE__, type);
  printf("[%s:%d] depth = %d\n", __FILE__, __LINE__, depth);
  printf("[%s:%d] interpolation = %d\n", __FILE__, __LINE__, interpolation);

#if defined HAVE_IPP && IPP_DISABLE_BLOCK
  // (OFF)
#endif

  RemapNNFunc nnfunc = 0;
  RemapFunc ifunc = 0;
  const void *ctab = 0;
  bool fixpt = depth == CV_8U;
  bool planar_input = false;

  if (interpolation == INTER_NEAREST) {
    nnfunc = nn_tab[depth];
    CV_Assert(nnfunc != 0);
  } else {
    if (interpolation == INTER_LINEAR) {
      printf("(cv trace) interpolation == INTER_LINEAR\n");
      ifunc = linear_tab[depth];
    } else if (interpolation == INTER_CUBIC){
      printf("(cv trace) interpolation == INTER_CUBIC\n");
      assert(0);
    //   ifunc = cubic_tab[depth];
    }
    else if (interpolation == INTER_LANCZOS4){
      printf("(cv trace) interpolation == INTER_LANCZOS4\n");
      assert(0);
    //   ifunc = lanczos4_tab[depth];
    }
    else
      CV_Error(CV_StsBadArg, "Unknown interpolation method");
    CV_Assert(ifunc != 0);
    ctab = initInterTab2D(interpolation, fixpt);
  }

  const cv::Mat *m1 = &map1, *m2 = &map2;

  if ((map1.type() == CV_16SC2 &&
       (map2.type() == CV_16UC1 || map2.type() == CV_16SC1 || map2.empty())) ||
      (map2.type() == CV_16SC2 &&
       (map1.type() == CV_16UC1 || map1.type() == CV_16SC1 || map1.empty()))) {
    if (map1.type() != CV_16SC2) {
      std::swap(m1, m2);
    }
  } else {
    CV_Assert(((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) &&
               map2.empty()) ||
              (map1.type() == CV_32FC1 && map2.type() == CV_32FC1));
    planar_input = map1.channels() == 1;
  }

  RemapInvoker invoker(src, dst, m1, m2, borderType, borderValue, planar_input,
                       nnfunc, ifunc, ctab);
  cv::parallel_for_(cv::Range(0, dst.rows), invoker, dst.total() / (double)(1 << 16));
}

class WarpPerspectiveInvoker : public cv::ParallelLoopBody {
public:
  WarpPerspectiveInvoker(const cv::Mat &_src, cv::Mat &_dst, const double *_M,
                         int _interpolation, int _borderType,
                         const cv::Scalar &_borderValue)
      : ParallelLoopBody(), src(_src), dst(_dst), M(_M),
        interpolation(_interpolation), borderType(_borderType),
        borderValue(_borderValue) {}

  virtual void operator()(const cv::Range &range) const {
    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ * BLOCK_SZ * 2], A[BLOCK_SZ * BLOCK_SZ];
    int x, y, x1, y1, width = dst.cols, height = dst.rows;

    int bh0 = std::min(BLOCK_SZ / 2, height);
    int bw0 = std::min(BLOCK_SZ * BLOCK_SZ / bh0, width);
    bh0 = std::min(BLOCK_SZ * BLOCK_SZ / bw0, height);

#if CV_SSE4_1
// (OFF)
#endif

    for (y = range.start; y < range.end; y += bh0) {
      for (x = 0; x < width; x += bw0) {
        int bw = std::min(bw0, width - x);
        int bh = std::min(bh0, range.end - y); // height

        cv::Mat _XY(bh, bw, CV_16SC2, XY), matA;
        cv::Mat dpart(dst, cv::Rect(x, y, bw, bh));

        for (y1 = 0; y1 < bh; y1++) {
          short *xy = XY + y1 * bw * 2;
          double X0 = M[0] * x + M[1] * (y + y1) + M[2];
          double Y0 = M[3] * x + M[4] * (y + y1) + M[5];
          double W0 = M[6] * x + M[7] * (y + y1) + M[8];

          if (interpolation == INTER_NEAREST) {
            x1 = 0;

#if CV_SSE4_1
// (OFF)
#endif

            for (; x1 < bw; x1++) {
              double W = W0 + M[6] * x1;
              W = W ? 1. / W : 0;
              double fX =
                  std::max((double)INT_MIN,
                           std::min((double)INT_MAX, (X0 + M[0] * x1) * W));
              double fY =
                  std::max((double)INT_MIN,
                           std::min((double)INT_MAX, (Y0 + M[3] * x1) * W));
              int X = cv::saturate_cast<int>(fX);
              int Y = cv::saturate_cast<int>(fY);

              xy[x1 * 2] = cv::saturate_cast<short>(X);
              xy[x1 * 2 + 1] = cv::saturate_cast<short>(Y);
            }
          } else {
            short *alpha = A + y1 * bw;
            x1 = 0;

#if CV_SSE4_1
// (OFF)
#endif

            for (; x1 < bw; x1++) {
              double W = W0 + M[6] * x1;
              W = W ? INTER_TAB_SIZE / W : 0;
              double fX =
                  std::max((double)INT_MIN,
                           std::min((double)INT_MAX, (X0 + M[0] * x1) * W));
              double fY =
                  std::max((double)INT_MIN,
                           std::min((double)INT_MAX, (Y0 + M[3] * x1) * W));
              int X = cv::saturate_cast<int>(fX);
              int Y = cv::saturate_cast<int>(fY);

              xy[x1 * 2] = cv::saturate_cast<short>(X >> INTER_BITS);
              xy[x1 * 2 + 1] = cv::saturate_cast<short>(Y >> INTER_BITS);
              alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                  (X & (INTER_TAB_SIZE - 1)));
            }
          }
        }

        printf("[%s:%d]\n", __FILE__, __LINE__);
        if (interpolation == INTER_NEAREST) {
          printf("case: INTER_NEAREST\n");
          remap(src, dpart, _XY, cv::Mat(), interpolation, borderType,
                borderValue);
        } else {
          printf("case: not INTER_NEAREST\n");
          cv::Mat _matA(bh, bw, CV_16U, A);
          remap(src, dpart, _XY, _matA, interpolation, borderType, borderValue);
        }
      }
    }
  }

private:
  cv::Mat src;
  cv::Mat dst;
  const double *M;
  int interpolation, borderType;
  cv::Scalar borderValue;
};

void warpPerspectve(int src_type, const uchar *src_data, size_t src_step,
                    int src_width, int src_height, uchar *dst_data,
                    size_t dst_step, int dst_width, int dst_height,
                    const double M[9], int interpolation, int borderType,
                    const double borderValue[4]) {
  /* [CV TRACE] Here! */
  printf("[%s:%d] hal::warpPerspective()\n", __FILE__, __LINE__);
  printf("src_type = %d\n", src_type);
  printf("borderType = %d\n", borderType);
  printf("borderValue[0] = %f\n", borderValue[0]);
  printf("borderValue[1] = %f\n", borderValue[1]);
  printf("borderValue[2] = %f\n", borderValue[2]);
  printf("borderValue[3] = %f\n", borderValue[3]);

  // CALL_HAL();

  cv::Mat src(cv::Size(src_width, src_height), src_type,
              const_cast<uchar *>(src_data), src_step);
  cv::Mat dst(cv::Size(dst_width, dst_height), src_type, dst_data, dst_step);

  cv::Range range(0, dst.rows);
  WarpPerspectiveInvoker invoker(src, dst, M, interpolation, borderType,
                                 cv::Scalar(borderValue[0], borderValue[1],
                                            borderValue[2], borderValue[3]));
  cv::parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

void warpPerspective(cv::InputArray _src, cv::OutputArray _dst,
                     cv::InputArray _M0, cv::Size dsize, int flags,
                     int borderType, const cv::Scalar &borderValue) {

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

  assert(_src.total() > 0);

  // CV_OCL_RUN()

  // CV_OCL_RUN()

  cv::Mat src = _src.getMat(), M0 = _M0.getMat();
  _dst.create(dsize.area() == 0 ? src.size() : dsize, src.type());
  cv::Mat dst = _dst.getMat();

  if (dst.data == src.data)
    src = src.clone();

  double M[9];
  cv::Mat matM(3, 3, CV_64F, M);
  int interpolation = flags & INTER_MAX;
  if (interpolation == INTER_AREA)
    interpolation = INTER_LINEAR;
  printf("interpolation = %d\n", interpolation);
  printf("borderType = %d\n", borderType);
  printf("borderValue[0] = %f\n", borderValue.val[0]);
  printf("borderValue[1] = %f\n", borderValue.val[1]);
  printf("borderValue[2] = %f\n", borderValue.val[2]);
  printf("borderValue[3] = %f\n", borderValue.val[3]);

  CV_Assert((M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 &&
            M0.cols == 3);
  M0.convertTo(matM, matM.type());

#if defined(HAVE_VIPP) && IPP_VERSION_X100 >= 810 && IPP_DISABLE_BLOCK
  // (OFF)
#endif

  if (!(flags & WARP_INVERSE_MAP)) {
    // [CV TEST] Here!
    invert(matM, matM);
  }

  warpPerspectve(src.type(), src.data, src.step, src.cols, src.rows, dst.data,
                 dst.step, dst.cols, dst.rows, matM.ptr<double>(),
                 interpolation, borderType, borderValue.val);
}

/* Calculates coefficients of perspective transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
 * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
 * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
 * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 *   cij - matrix coefficients, c22 = 1
 */
cv::Mat getPerspectiveTransform( const cv::Point2f src[], const cv::Point2f dst[] )
{
    cv::Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
    double a[8][8], b[8];
    cv::Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    for( int i = 0; i < 4; ++i )
    {
        a[i][0] = a[i+4][3] = src[i].x;
        a[i][1] = a[i+4][4] = src[i].y;
        a[i][2] = a[i+4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] =
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
        a[i][6] = -src[i].x*dst[i].x;
        a[i][7] = -src[i].y*dst[i].x;
        a[i+4][6] = -src[i].x*dst[i].y;
        a[i+4][7] = -src[i].y*dst[i].y;
        b[i] = dst[i].x;
        b[i+4] = dst[i].y;
    }

    cv::solve( A, B, X, cv::DECOMP_SVD );
    M.ptr<double>()[8] = 1.;

    return M;
}

/* Calculates coefficients of affine transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 | |c01| |u1|
 * | x2 y2  1  0  0  0 | |c02| |u2|
 * |  0  0  0 x0 y0  1 | |c10| |v0|
 * |  0  0  0 x1 y1  1 | |c11| |v1|
 * \  0  0  0 x2 y2  1 / |c12| |v2|
 *
 * where:
 *   cij - matrix coefficients
 */

cv::Mat getAffineTransform( const cv::Point2f src[], const cv::Point2f dst[] )
{
    cv::Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.ptr());
    double a[6*6], b[6];
    cv::Mat A(6, 6, CV_64F, a), B(6, 1, CV_64F, b);

    for( int i = 0; i < 3; i++ )
    {
        int j = i*12;
        int k = i*12+6;
        a[j] = a[k+3] = src[i].x;
        a[j+1] = a[k+4] = src[i].y;
        a[j+2] = a[k+5] = 1;
        a[j+3] = a[j+4] = a[j+5] = 0;
        a[k] = a[k+1] = a[k+2] = 0;
        b[i*2] = dst[i].x;
        b[i*2+1] = dst[i].y;
    }

    cv::solve( A, B, X );
    return M;
}

cv::Mat getPerspectiveTransform(cv::InputArray _src, cv::InputArray _dst)
{
    cv::Mat src = _src.getMat(), dst = _dst.getMat();
    CV_Assert(src.checkVector(2, CV_32F) == 4 && dst.checkVector(2, CV_32F) == 4);
    return getPerspectiveTransform((const cv::Point2f*)src.data, (const cv::Point2f*)dst.data);
}

cv::Mat getAffineTransform(cv::InputArray _src, cv::InputArray _dst)
{
    cv::Mat src = _src.getMat(), dst = _dst.getMat();
    CV_Assert(src.checkVector(2, CV_32F) == 3 && dst.checkVector(2, CV_32F) == 3);
    return getAffineTransform((const cv::Point2f*)src.data, (const cv::Point2f*)dst.data);
}

void ai::getPerspectiveTransform() {
  printf("ai::getPerspectiveTransform\n");
}
void ai::getAffineTransform() {
  printf("ai::getAffineTransform\n");
}
void ai::warpPerspective() {
  printf("ai::warpPerspective\n");
}
void ai::warpAffine() {
  printf("ai::warpAffine\n");
}