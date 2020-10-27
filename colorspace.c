//
// Created by von on 2020/8/17.
//

// Kr and Kb YUV<-->RGB constants.
constexpr double FCC_KR = 0.3;
constexpr double FCC_KB = 0.11;
constexpr double SMPTE_240M_KR = 0.212;
constexpr double SMPTE_240M_KB = 0.087;
constexpr double REC_601_KR = 0.299;
constexpr double REC_601_KB = 0.114;
constexpr double REC_709_KR = 0.2126;
constexpr double REC_709_KB = 0.0722;
constexpr double REC_2020_KR = 0.2627;
constexpr double REC_2020_KB = 0.0593;

enum MatrixCoefficients {
        UNSPECIFIED,
        RGB,
        REC_601,
        REC_709,
        FCC,
        SMPTE_240M,
        YCGCO,
        REC_2020_NCL,
        REC_2020_CL,
        CHROMATICITY_DERIVED_NCL,
        CHROMATICITY_DERIVED_CL,
        REC_2100_LMS,
        REC_2100_ICTCP,
};

/**
 * Fixed size 3x3 matrix.
 */
typedef double (*Matrix3x3)[3] ;

Matrix3x3 ncl_rgb_to_yuv_matrix_from_kr_kb(double kr, double kb)
{
    double ret[3][3];
    double kg = 1.0 - kr - kb;
    double uscale;
    double vscale;

    uscale = 1.0 / (2.0 - 2.0 * kb);
    vscale = 1.0 / (2.0 - 2.0 * kr);

    ret[0][0] = kr;
    ret[0][1] = kg;
    ret[0][2] = kb;

    ret[1][0] = -kr * uscale;
    ret[1][1] = -kg * uscale;
    ret[1][2] = (1.0 - kb) * uscale;

    ret[2][0] = (1.0 - kr) * vscale;
    ret[2][1] = -kg * vscale;
    ret[2][2] = -kb * vscale;

    return ret;
}

Matrix3x3 ncl_rgb_to_yuv_matrix(MatrixCoefficients matrix)
{
    double kr, kb;

    switch (matrix)
    {
        case MatrixCoefficients::YCGCO:
            return {
                    {  0.25, 0.5,  0.25 },
                    { -0.25, 0.5, -0.25 },
                    {  0.5,  0,   -0.5 }
            };
        case MatrixCoefficients::REC_2100_LMS:
            return {
                    { 1688.0 / 4096.0, 2146.0 / 4096.0,  262.0 / 4096.0 },
                    {  683.0 / 4096.0, 2951.0 / 4096.0,  462.0 / 4096.0 },
                    {   99.0 / 4096.0,  309.0 / 4096.0, 3688.0 / 4096.0 }
            };
        default:
            get_yuv_constants(&kr, &kb, matrix);
            return ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb);
    }
}

Matrix3x3 ncl_yuv_to_rgb_matrix(MatrixCoefficients matrix)
{
    return inverse(ncl_rgb_to_yuv_matrix(matrix));
}

/////
double det2(double a00, double a01, double a10, double a11)
{
    return a00 * a11 - a01 * a10;
}


double determinant(const Matrix3x3 &m) noexcept
{
    double det = 0;

    det += m[0][0] * det2(m[1][1], m[1][2], m[2][1], m[2][2]);
    det -= m[0][1] * det2(m[1][0], m[1][2], m[2][0], m[2][2]);
    det += m[0][2] * det2(m[1][0], m[1][1], m[2][0], m[2][1]);

    return det;
}

Matrix3x3 inverse(const Matrix3x3 &m) noexcept
{
    double ret[3][3];
    double det = determinant(m);

    ret[0][0] = det2(m[1][1], m[1][2], m[2][1], m[2][2]) / det;
    ret[0][1] = det2(m[0][2], m[0][1], m[2][2], m[2][1]) / det;
    ret[0][2] = det2(m[0][1], m[0][2], m[1][1], m[1][2]) / det;
    ret[1][0] = det2(m[1][2], m[1][0], m[2][2], m[2][0]) / det;
    ret[1][1] = det2(m[0][0], m[0][2], m[2][0], m[2][2]) / det;
    ret[1][2] = det2(m[0][2], m[0][0], m[1][2], m[1][0]) / det;
    ret[2][0] = det2(m[1][0], m[1][1], m[2][0], m[2][1]) / det;
    ret[2][1] = det2(m[0][1], m[0][0], m[2][1], m[2][0]) / det;
    ret[2][2] = det2(m[0][0], m[0][1], m[1][0], m[1][1]) / det;

    return ret;
}

int main(int argc, char **argv) {
    // double a[3][3] = {{1,2,3}, {3,4,5}, {5,6,7}};
    Matrix3x3 a = ncl_yuv_to_rgb_matrix(REC_709);
    return 0;
}