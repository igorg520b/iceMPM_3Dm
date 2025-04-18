#ifndef HELPER_MATH_CUH
#define HELPER_MATH_CUH

//#include "givens.cuh"

//#include "parameters_sim_3d.h"



#define gone 1065353216
#define gsine_pi_over_eight 1053028117
#define gcosine_pi_over_eight 1064076127
#define gone_half 0.5f
#define gsmall_number 1.e-12f
#define gtiny_number 1.e-20f
#define gfour_gamma_squared 5.8284273147583007813f

union un {
    float f;
    unsigned int ui;
};

template<typename T>
__forceinline__ __device__ void svd(
    T a11,
    T a12,
    T a13,
    T a21,
    T a22,
    T a23,
    T a31,
    T a32,
    T a33,// input A
    T& u11,
    T& u12,
    T& u13,
    T& u21,
    T& u22,
    T& u23,
    T& u31,
    T& u32,
    T& u33,// output U
    T& s11,
    //float &s12, float &s13, float &s21,
    T& s22,
    //float &s23, float &s31, float &s32,
    T& s33,// output S
    T& v11,
    T& v12,
    T& v13,
    T& v21,
    T& v22,
    T& v23,
    T& v31,
    T& v32,
    T& v33// output V
    ) {
    un s_a11;
    un s_a21;
    un s_a31;
    un s_a12;
    un s_a22;
    un s_a32;
    un s_a13;
    un s_a23;
    un s_a33;
    un s_u11;
    un s_u21;
    un s_u31;
    un s_u12;
    un s_u22;
    un s_u32;
    un s_u13;
    un s_u23;
    un s_u33;
    un s_v11;
    un s_v21;
    un s_v31;
    un s_v12;
    un s_v22;
    un s_v32;
    un s_v13;
    un s_v23;
    un s_v33;
    un s_c;
    un s_s;
    un s_ch;
    un s_sh;
    un s_tmp1;
    un s_tmp2;
    un s_tmp3;
    un s_tmp4;
    un s_tmp5;
    un s_s11;
    un s_s21;
    un s_s31;
    un s_s22;
    un s_s32;
    un s_s33;
    un s_qvs;
    un s_qvvx;
    un s_qvvy;
    un s_qvvz;

    s_a11.f = a11;
    s_a12.f = a12;
    s_a13.f = a13;
    s_a21.f = a21;
    s_a22.f = a22;
    s_a23.f = a23;
    s_a31.f = a31;
    s_a32.f = a32;
    s_a33.f = a33;

    //###########################################################
    // Compute normal equations matrix
    //###########################################################

    s_s11.f	 = s_a11.f * s_a11.f;
    s_tmp1.f = s_a21.f * s_a21.f;
    s_s11.f	 = __fadd_rn(s_tmp1.f, s_s11.f);
    s_tmp1.f = s_a31.f * s_a31.f;
    s_s11.f	 = __fadd_rn(s_tmp1.f, s_s11.f);

    s_s21.f	 = s_a12.f * s_a11.f;
    s_tmp1.f = s_a22.f * s_a21.f;
    s_s21.f	 = __fadd_rn(s_tmp1.f, s_s21.f);
    s_tmp1.f = s_a32.f * s_a31.f;
    s_s21.f	 = __fadd_rn(s_tmp1.f, s_s21.f);

    s_s31.f	 = s_a13.f * s_a11.f;
    s_tmp1.f = s_a23.f * s_a21.f;
    s_s31.f	 = __fadd_rn(s_tmp1.f, s_s31.f);
    s_tmp1.f = s_a33.f * s_a31.f;
    s_s31.f	 = __fadd_rn(s_tmp1.f, s_s31.f);

    s_s22.f	 = s_a12.f * s_a12.f;
    s_tmp1.f = s_a22.f * s_a22.f;
    s_s22.f	 = __fadd_rn(s_tmp1.f, s_s22.f);
    s_tmp1.f = s_a32.f * s_a32.f;
    s_s22.f	 = __fadd_rn(s_tmp1.f, s_s22.f);

    s_s32.f	 = s_a13.f * s_a12.f;
    s_tmp1.f = s_a23.f * s_a22.f;
    s_s32.f	 = __fadd_rn(s_tmp1.f, s_s32.f);
    s_tmp1.f = s_a33.f * s_a32.f;
    s_s32.f	 = __fadd_rn(s_tmp1.f, s_s32.f);

    s_s33.f	 = s_a13.f * s_a13.f;
    s_tmp1.f = s_a23.f * s_a23.f;
    s_s33.f	 = __fadd_rn(s_tmp1.f, s_s33.f);
    s_tmp1.f = s_a33.f * s_a33.f;
    s_s33.f	 = __fadd_rn(s_tmp1.f, s_s33.f);

    s_qvs.f	 = 1.f;
    s_qvvx.f = 0.f;
    s_qvvy.f = 0.f;
    s_qvvz.f = 0.f;

    //###########################################################
    // Solve symmetric eigenproblem using Jacobi iteration
    //###########################################################
    for(int i = 0; i < 4; i++) {
        s_sh.f	 = s_s21.f * 0.5f;
        s_tmp5.f = __fsub_rn(s_s11.f, s_s22.f);

        s_tmp2.f  = s_sh.f * s_sh.f;
        s_tmp1.ui = (s_tmp2.f >= gtiny_number) ? 0xffffffff : 0;
        s_sh.ui	  = s_tmp1.ui & s_sh.ui;
        s_ch.ui	  = s_tmp1.ui & s_tmp5.ui;
        s_tmp2.ui = ~s_tmp1.ui & gone;
        s_ch.ui	  = s_ch.ui | s_tmp2.ui;

        s_tmp1.f = s_sh.f * s_sh.f;
        s_tmp2.f = s_ch.f * s_ch.f;
        s_tmp3.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
        s_tmp4.f = __frsqrt_rn(s_tmp3.f);

        s_sh.f	  = s_tmp4.f * s_sh.f;
        s_ch.f	  = s_tmp4.f * s_ch.f;
        s_tmp1.f  = gfour_gamma_squared * s_tmp1.f;
        s_tmp1.ui = (s_tmp2.f <= s_tmp1.f) ? 0xffffffff : 0;

        s_tmp2.ui = gsine_pi_over_eight & s_tmp1.ui;
        s_sh.ui	  = ~s_tmp1.ui & s_sh.ui;
        s_sh.ui	  = s_sh.ui | s_tmp2.ui;
        s_tmp2.ui = gcosine_pi_over_eight & s_tmp1.ui;
        s_ch.ui	  = ~s_tmp1.ui & s_ch.ui;
        s_ch.ui	  = s_ch.ui | s_tmp2.ui;

        s_tmp1.f = s_sh.f * s_sh.f;
        s_tmp2.f = s_ch.f * s_ch.f;
        s_c.f	 = __fsub_rn(s_tmp2.f, s_tmp1.f);
        s_s.f	 = s_ch.f * s_sh.f;
        s_s.f	 = __fadd_rn(s_s.f, s_s.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", s_s.f, s_c.f, s_sh.f, s_ch.f);
#endif \
    //########################################################### \
    // Perform the actual Givens conjugation \
    //###########################################################

        s_tmp3.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
        s_s33.f	 = s_s33.f * s_tmp3.f;
        s_s31.f	 = s_s31.f * s_tmp3.f;
        s_s32.f	 = s_s32.f * s_tmp3.f;
        s_s33.f	 = s_s33.f * s_tmp3.f;

        s_tmp1.f = s_s.f * s_s31.f;
        s_tmp2.f = s_s.f * s_s32.f;
        s_s31.f	 = s_c.f * s_s31.f;
        s_s32.f	 = s_c.f * s_s32.f;
        s_s31.f	 = __fadd_rn(s_tmp2.f, s_s31.f);
        s_s32.f	 = __fsub_rn(s_s32.f, s_tmp1.f);

        s_tmp2.f = s_s.f * s_s.f;
        s_tmp1.f = s_s22.f * s_tmp2.f;
        s_tmp3.f = s_s11.f * s_tmp2.f;
        s_tmp4.f = s_c.f * s_c.f;
        s_s11.f	 = s_s11.f * s_tmp4.f;
        s_s22.f	 = s_s22.f * s_tmp4.f;
        s_s11.f	 = __fadd_rn(s_s11.f, s_tmp1.f);
        s_s22.f	 = __fadd_rn(s_s22.f, s_tmp3.f);
        s_tmp4.f = __fsub_rn(s_tmp4.f, s_tmp2.f);
        s_tmp2.f = __fadd_rn(s_s21.f, s_s21.f);
        s_s21.f	 = s_s21.f * s_tmp4.f;
        s_tmp4.f = s_c.f * s_s.f;
        s_tmp2.f = s_tmp2.f * s_tmp4.f;
        s_tmp5.f = s_tmp5.f * s_tmp4.f;
        s_s11.f	 = __fadd_rn(s_s11.f, s_tmp2.f);
        s_s21.f	 = __fsub_rn(s_s21.f, s_tmp5.f);
        s_s22.f	 = __fsub_rn(s_s22.f, s_tmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", s_s11.f);
        printf("%.20g %.20g\n", s_s21.f, s_s22.f);
        printf("%.20g %.20g %.20g\n", s_s31.f, s_s32.f, s_s33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        s_tmp1.f = s_sh.f * s_qvvx.f;
        s_tmp2.f = s_sh.f * s_qvvy.f;
        s_tmp3.f = s_sh.f * s_qvvz.f;
        s_sh.f	 = s_sh.f * s_qvs.f;

        s_qvs.f	 = s_ch.f * s_qvs.f;
        s_qvvx.f = s_ch.f * s_qvvx.f;
        s_qvvy.f = s_ch.f * s_qvvy.f;
        s_qvvz.f = s_ch.f * s_qvvz.f;

        s_qvvz.f = __fadd_rn(s_qvvz.f, s_sh.f);
        s_qvs.f	 = __fsub_rn(s_qvs.f, s_tmp3.f);
        s_qvvx.f = __fadd_rn(s_qvvx.f, s_tmp2.f);
        s_qvvy.f = __fsub_rn(s_qvvy.f, s_tmp1.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU q %.20g %.20g %.20g %.20g\n", s_qvvx.f, s_qvvy.f, s_qvvz.f, s_qvs.f);
#endif

        //////////////////////////////////////////////////////////////////////////
        // (1->3)
        //////////////////////////////////////////////////////////////////////////
        s_sh.f	 = s_s32.f * 0.5f;
        s_tmp5.f = __fsub_rn(s_s22.f, s_s33.f);

        s_tmp2.f  = s_sh.f * s_sh.f;
        s_tmp1.ui = (s_tmp2.f >= gtiny_number) ? 0xffffffff : 0;
        s_sh.ui	  = s_tmp1.ui & s_sh.ui;
        s_ch.ui	  = s_tmp1.ui & s_tmp5.ui;
        s_tmp2.ui = ~s_tmp1.ui & gone;
        s_ch.ui	  = s_ch.ui | s_tmp2.ui;

        s_tmp1.f = s_sh.f * s_sh.f;
        s_tmp2.f = s_ch.f * s_ch.f;
        s_tmp3.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
        s_tmp4.f = __frsqrt_rn(s_tmp3.f);

        s_sh.f	  = s_tmp4.f * s_sh.f;
        s_ch.f	  = s_tmp4.f * s_ch.f;
        s_tmp1.f  = gfour_gamma_squared * s_tmp1.f;
        s_tmp1.ui = (s_tmp2.f <= s_tmp1.f) ? 0xffffffff : 0;

        s_tmp2.ui = gsine_pi_over_eight & s_tmp1.ui;
        s_sh.ui	  = ~s_tmp1.ui & s_sh.ui;
        s_sh.ui	  = s_sh.ui | s_tmp2.ui;
        s_tmp2.ui = gcosine_pi_over_eight & s_tmp1.ui;
        s_ch.ui	  = ~s_tmp1.ui & s_ch.ui;
        s_ch.ui	  = s_ch.ui | s_tmp2.ui;

        s_tmp1.f = s_sh.f * s_sh.f;
        s_tmp2.f = s_ch.f * s_ch.f;
        s_c.f	 = __fsub_rn(s_tmp2.f, s_tmp1.f);
        s_s.f	 = s_ch.f * s_sh.f;
        s_s.f	 = __fadd_rn(s_s.f, s_s.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", s_s.f, s_c.f, s_sh.f, s_ch.f);
#endif

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        s_tmp3.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
        s_s11.f	 = s_s11.f * s_tmp3.f;
        s_s21.f	 = s_s21.f * s_tmp3.f;
        s_s31.f	 = s_s31.f * s_tmp3.f;
        s_s11.f	 = s_s11.f * s_tmp3.f;

        s_tmp1.f = s_s.f * s_s21.f;
        s_tmp2.f = s_s.f * s_s31.f;
        s_s21.f	 = s_c.f * s_s21.f;
        s_s31.f	 = s_c.f * s_s31.f;
        s_s21.f	 = __fadd_rn(s_tmp2.f, s_s21.f);
        s_s31.f	 = __fsub_rn(s_s31.f, s_tmp1.f);

        s_tmp2.f = s_s.f * s_s.f;
        s_tmp1.f = s_s33.f * s_tmp2.f;
        s_tmp3.f = s_s22.f * s_tmp2.f;
        s_tmp4.f = s_c.f * s_c.f;
        s_s22.f	 = s_s22.f * s_tmp4.f;
        s_s33.f	 = s_s33.f * s_tmp4.f;
        s_s22.f	 = __fadd_rn(s_s22.f, s_tmp1.f);
        s_s33.f	 = __fadd_rn(s_s33.f, s_tmp3.f);
        s_tmp4.f = __fsub_rn(s_tmp4.f, s_tmp2.f);
        s_tmp2.f = __fadd_rn(s_s32.f, s_s32.f);
        s_s32.f	 = s_s32.f * s_tmp4.f;
        s_tmp4.f = s_c.f * s_s.f;
        s_tmp2.f = s_tmp2.f * s_tmp4.f;
        s_tmp5.f = s_tmp5.f * s_tmp4.f;
        s_s22.f	 = __fadd_rn(s_s22.f, s_tmp2.f);
        s_s32.f	 = __fsub_rn(s_s32.f, s_tmp5.f);
        s_s33.f	 = __fsub_rn(s_s33.f, s_tmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", s_s11.f);
        printf("%.20g %.20g\n", s_s21.f, s_s22.f);
        printf("%.20g %.20g %.20g\n", s_s31.f, s_s32.f, s_s33.f);
#endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        s_tmp1.f = s_sh.f * s_qvvx.f;
        s_tmp2.f = s_sh.f * s_qvvy.f;
        s_tmp3.f = s_sh.f * s_qvvz.f;
        s_sh.f	 = s_sh.f * s_qvs.f;

        s_qvs.f	 = s_ch.f * s_qvs.f;
        s_qvvx.f = s_ch.f * s_qvvx.f;
        s_qvvy.f = s_ch.f * s_qvvy.f;
        s_qvvz.f = s_ch.f * s_qvvz.f;

        s_qvvx.f = __fadd_rn(s_qvvx.f, s_sh.f);
        s_qvs.f	 = __fsub_rn(s_qvs.f, s_tmp1.f);
        s_qvvy.f = __fadd_rn(s_qvvy.f, s_tmp3.f);
        s_qvvz.f = __fsub_rn(s_qvvz.f, s_tmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU q %.20g %.20g %.20g %.20g\n", s_qvvx.f, s_qvvy.f, s_qvvz.f, s_qvs.f);
#endif
#if 1 \
    ////////////////////////////////////////////////////////////////////////// \
    // 1 -> 2 \
    //////////////////////////////////////////////////////////////////////////

        s_sh.f	 = s_s31.f * 0.5f;
        s_tmp5.f = __fsub_rn(s_s33.f, s_s11.f);

        s_tmp2.f  = s_sh.f * s_sh.f;
        s_tmp1.ui = (s_tmp2.f >= gtiny_number) ? 0xffffffff : 0;
        s_sh.ui	  = s_tmp1.ui & s_sh.ui;
        s_ch.ui	  = s_tmp1.ui & s_tmp5.ui;
        s_tmp2.ui = ~s_tmp1.ui & gone;
        s_ch.ui	  = s_ch.ui | s_tmp2.ui;

        s_tmp1.f = s_sh.f * s_sh.f;
        s_tmp2.f = s_ch.f * s_ch.f;
        s_tmp3.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
        s_tmp4.f = __frsqrt_rn(s_tmp3.f);

        s_sh.f	  = s_tmp4.f * s_sh.f;
        s_ch.f	  = s_tmp4.f * s_ch.f;
        s_tmp1.f  = gfour_gamma_squared * s_tmp1.f;
        s_tmp1.ui = (s_tmp2.f <= s_tmp1.f) ? 0xffffffff : 0;

        s_tmp2.ui = gsine_pi_over_eight & s_tmp1.ui;
        s_sh.ui	  = ~s_tmp1.ui & s_sh.ui;
        s_sh.ui	  = s_sh.ui | s_tmp2.ui;
        s_tmp2.ui = gcosine_pi_over_eight & s_tmp1.ui;
        s_ch.ui	  = ~s_tmp1.ui & s_ch.ui;
        s_ch.ui	  = s_ch.ui | s_tmp2.ui;

        s_tmp1.f = s_sh.f * s_sh.f;
        s_tmp2.f = s_ch.f * s_ch.f;
        s_c.f	 = __fsub_rn(s_tmp2.f, s_tmp1.f);
        s_s.f	 = s_ch.f * s_sh.f;
        s_s.f	 = __fadd_rn(s_s.f, s_s.f);

#	ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", s_s.f, s_c.f, s_sh.f, s_ch.f);
#	endif

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        s_tmp3.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
        s_s22.f	 = s_s22.f * s_tmp3.f;
        s_s32.f	 = s_s32.f * s_tmp3.f;
        s_s21.f	 = s_s21.f * s_tmp3.f;
        s_s22.f	 = s_s22.f * s_tmp3.f;

        s_tmp1.f = s_s.f * s_s32.f;
        s_tmp2.f = s_s.f * s_s21.f;
        s_s32.f	 = s_c.f * s_s32.f;
        s_s21.f	 = s_c.f * s_s21.f;
        s_s32.f	 = __fadd_rn(s_tmp2.f, s_s32.f);
        s_s21.f	 = __fsub_rn(s_s21.f, s_tmp1.f);

        s_tmp2.f = s_s.f * s_s.f;
        s_tmp1.f = s_s11.f * s_tmp2.f;
        s_tmp3.f = s_s33.f * s_tmp2.f;
        s_tmp4.f = s_c.f * s_c.f;
        s_s33.f	 = s_s33.f * s_tmp4.f;
        s_s11.f	 = s_s11.f * s_tmp4.f;
        s_s33.f	 = __fadd_rn(s_s33.f, s_tmp1.f);
        s_s11.f	 = __fadd_rn(s_s11.f, s_tmp3.f);
        s_tmp4.f = __fsub_rn(s_tmp4.f, s_tmp2.f);
        s_tmp2.f = __fadd_rn(s_s31.f, s_s31.f);
        s_s31.f	 = s_s31.f * s_tmp4.f;
        s_tmp4.f = s_c.f * s_s.f;
        s_tmp2.f = s_tmp2.f * s_tmp4.f;
        s_tmp5.f = s_tmp5.f * s_tmp4.f;
        s_s33.f	 = __fadd_rn(s_s33.f, s_tmp2.f);
        s_s31.f	 = __fsub_rn(s_s31.f, s_tmp5.f);
        s_s11.f	 = __fsub_rn(s_s11.f, s_tmp2.f);

#	ifdef DEBUG_JACOBI_CONJUGATE
        printf("%.20g\n", s_s11.f);
        printf("%.20g %.20g\n", s_s21.f, s_s22.f);
        printf("%.20g %.20g %.20g\n", s_s31.f, s_s32.f, s_s33.f);
#	endif

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        s_tmp1.f = s_sh.f * s_qvvx.f;
        s_tmp2.f = s_sh.f * s_qvvy.f;
        s_tmp3.f = s_sh.f * s_qvvz.f;
        s_sh.f	 = s_sh.f * s_qvs.f;

        s_qvs.f	 = s_ch.f * s_qvs.f;
        s_qvvx.f = s_ch.f * s_qvvx.f;
        s_qvvy.f = s_ch.f * s_qvvy.f;
        s_qvvz.f = s_ch.f * s_qvvz.f;

        s_qvvy.f = __fadd_rn(s_qvvy.f, s_sh.f);
        s_qvs.f	 = __fsub_rn(s_qvs.f, s_tmp2.f);
        s_qvvz.f = __fadd_rn(s_qvvz.f, s_tmp1.f);
        s_qvvx.f = __fsub_rn(s_qvvx.f, s_tmp3.f);
#endif
    }

    //###########################################################
    // Normalize quaternion for matrix V
    //###########################################################

    s_tmp2.f = s_qvs.f * s_qvs.f;
    s_tmp1.f = s_qvvx.f * s_qvvx.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = s_qvvy.f * s_qvvy.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = s_qvvz.f * s_qvvz.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);

    s_tmp1.f = __frsqrt_rn(s_tmp2.f);
    s_tmp4.f = s_tmp1.f * 0.5f;
    s_tmp3.f = s_tmp1.f * s_tmp4.f;
    s_tmp3.f = s_tmp1.f * s_tmp3.f;
    s_tmp3.f = s_tmp2.f * s_tmp3.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp1.f = __fsub_rn(s_tmp1.f, s_tmp3.f);

    s_qvs.f	 = s_qvs.f * s_tmp1.f;
    s_qvvx.f = s_qvvx.f * s_tmp1.f;
    s_qvvy.f = s_qvvy.f * s_tmp1.f;
    s_qvvz.f = s_qvvz.f * s_tmp1.f;

    //###########################################################
    // Transform quaternion to matrix V
    //###########################################################

    s_tmp1.f = s_qvvx.f * s_qvvx.f;
    s_tmp2.f = s_qvvy.f * s_qvvy.f;
    s_tmp3.f = s_qvvz.f * s_qvvz.f;
    s_v11.f	 = s_qvs.f * s_qvs.f;
    s_v22.f	 = __fsub_rn(s_v11.f, s_tmp1.f);
    s_v33.f	 = __fsub_rn(s_v22.f, s_tmp2.f);
    s_v33.f	 = __fadd_rn(s_v33.f, s_tmp3.f);
    s_v22.f	 = __fadd_rn(s_v22.f, s_tmp2.f);
    s_v22.f	 = __fsub_rn(s_v22.f, s_tmp3.f);
    s_v11.f	 = __fadd_rn(s_v11.f, s_tmp1.f);
    s_v11.f	 = __fsub_rn(s_v11.f, s_tmp2.f);
    s_v11.f	 = __fsub_rn(s_v11.f, s_tmp3.f);
    s_tmp1.f = __fadd_rn(s_qvvx.f, s_qvvx.f);
    s_tmp2.f = __fadd_rn(s_qvvy.f, s_qvvy.f);
    s_tmp3.f = __fadd_rn(s_qvvz.f, s_qvvz.f);
    s_v32.f	 = s_qvs.f * s_tmp1.f;
    s_v13.f	 = s_qvs.f * s_tmp2.f;
    s_v21.f	 = s_qvs.f * s_tmp3.f;
    s_tmp1.f = s_qvvy.f * s_tmp1.f;
    s_tmp2.f = s_qvvz.f * s_tmp2.f;
    s_tmp3.f = s_qvvx.f * s_tmp3.f;
    s_v12.f	 = __fsub_rn(s_tmp1.f, s_v21.f);
    s_v23.f	 = __fsub_rn(s_tmp2.f, s_v32.f);
    s_v31.f	 = __fsub_rn(s_tmp3.f, s_v13.f);
    s_v21.f	 = __fadd_rn(s_tmp1.f, s_v21.f);
    s_v32.f	 = __fadd_rn(s_tmp2.f, s_v32.f);
    s_v13.f	 = __fadd_rn(s_tmp3.f, s_v13.f);

    ///###########################################################
    // Multiply (from the right) with V
    //###########################################################

    s_tmp2.f = s_a12.f;
    s_tmp3.f = s_a13.f;
    s_a12.f	 = s_v12.f * s_a11.f;
    s_a13.f	 = s_v13.f * s_a11.f;
    s_a11.f	 = s_v11.f * s_a11.f;
    s_tmp1.f = s_v21.f * s_tmp2.f;
    s_a11.f	 = __fadd_rn(s_a11.f, s_tmp1.f);
    s_tmp1.f = s_v31.f * s_tmp3.f;
    s_a11.f	 = __fadd_rn(s_a11.f, s_tmp1.f);
    s_tmp1.f = s_v22.f * s_tmp2.f;
    s_a12.f	 = __fadd_rn(s_a12.f, s_tmp1.f);
    s_tmp1.f = s_v32.f * s_tmp3.f;
    s_a12.f	 = __fadd_rn(s_a12.f, s_tmp1.f);
    s_tmp1.f = s_v23.f * s_tmp2.f;
    s_a13.f	 = __fadd_rn(s_a13.f, s_tmp1.f);
    s_tmp1.f = s_v33.f * s_tmp3.f;
    s_a13.f	 = __fadd_rn(s_a13.f, s_tmp1.f);

    s_tmp2.f = s_a22.f;
    s_tmp3.f = s_a23.f;
    s_a22.f	 = s_v12.f * s_a21.f;
    s_a23.f	 = s_v13.f * s_a21.f;
    s_a21.f	 = s_v11.f * s_a21.f;
    s_tmp1.f = s_v21.f * s_tmp2.f;
    s_a21.f	 = __fadd_rn(s_a21.f, s_tmp1.f);
    s_tmp1.f = s_v31.f * s_tmp3.f;
    s_a21.f	 = __fadd_rn(s_a21.f, s_tmp1.f);
    s_tmp1.f = s_v22.f * s_tmp2.f;
    s_a22.f	 = __fadd_rn(s_a22.f, s_tmp1.f);
    s_tmp1.f = s_v32.f * s_tmp3.f;
    s_a22.f	 = __fadd_rn(s_a22.f, s_tmp1.f);
    s_tmp1.f = s_v23.f * s_tmp2.f;
    s_a23.f	 = __fadd_rn(s_a23.f, s_tmp1.f);
    s_tmp1.f = s_v33.f * s_tmp3.f;
    s_a23.f	 = __fadd_rn(s_a23.f, s_tmp1.f);

    s_tmp2.f = s_a32.f;
    s_tmp3.f = s_a33.f;
    s_a32.f	 = s_v12.f * s_a31.f;
    s_a33.f	 = s_v13.f * s_a31.f;
    s_a31.f	 = s_v11.f * s_a31.f;
    s_tmp1.f = s_v21.f * s_tmp2.f;
    s_a31.f	 = __fadd_rn(s_a31.f, s_tmp1.f);
    s_tmp1.f = s_v31.f * s_tmp3.f;
    s_a31.f	 = __fadd_rn(s_a31.f, s_tmp1.f);
    s_tmp1.f = s_v22.f * s_tmp2.f;
    s_a32.f	 = __fadd_rn(s_a32.f, s_tmp1.f);
    s_tmp1.f = s_v32.f * s_tmp3.f;
    s_a32.f	 = __fadd_rn(s_a32.f, s_tmp1.f);
    s_tmp1.f = s_v23.f * s_tmp2.f;
    s_a33.f	 = __fadd_rn(s_a33.f, s_tmp1.f);
    s_tmp1.f = s_v33.f * s_tmp3.f;
    s_a33.f	 = __fadd_rn(s_a33.f, s_tmp1.f);

    //###########################################################
    // Permute columns such that the singular values are sorted
    //###########################################################

    s_tmp1.f = s_a11.f * s_a11.f;
    s_tmp4.f = s_a21.f * s_a21.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp4.f = s_a31.f * s_a31.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);

    s_tmp2.f = s_a12.f * s_a12.f;
    s_tmp4.f = s_a22.f * s_a22.f;
    s_tmp2.f = __fadd_rn(s_tmp2.f, s_tmp4.f);
    s_tmp4.f = s_a32.f * s_a32.f;
    s_tmp2.f = __fadd_rn(s_tmp2.f, s_tmp4.f);

    s_tmp3.f = s_a13.f * s_a13.f;
    s_tmp4.f = s_a23.f * s_a23.f;
    s_tmp3.f = __fadd_rn(s_tmp3.f, s_tmp4.f);
    s_tmp4.f = s_a33.f * s_a33.f;
    s_tmp3.f = __fadd_rn(s_tmp3.f, s_tmp4.f);

    // Swap columns 1-2 if necessary

    s_tmp4.ui = (s_tmp1.f < s_tmp2.f) ? 0xffffffff : 0;
    s_tmp5.ui = s_a11.ui ^ s_a12.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a11.ui  = s_a11.ui ^ s_tmp5.ui;
    s_a12.ui  = s_a12.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_a21.ui ^ s_a22.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a21.ui  = s_a21.ui ^ s_tmp5.ui;
    s_a22.ui  = s_a22.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_a31.ui ^ s_a32.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a31.ui  = s_a31.ui ^ s_tmp5.ui;
    s_a32.ui  = s_a32.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v11.ui ^ s_v12.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v11.ui  = s_v11.ui ^ s_tmp5.ui;
    s_v12.ui  = s_v12.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v21.ui ^ s_v22.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v21.ui  = s_v21.ui ^ s_tmp5.ui;
    s_v22.ui  = s_v22.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v31.ui ^ s_v32.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v31.ui  = s_v31.ui ^ s_tmp5.ui;
    s_v32.ui  = s_v32.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_tmp1.ui ^ s_tmp2.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_tmp1.ui = s_tmp1.ui ^ s_tmp5.ui;
    s_tmp2.ui = s_tmp2.ui ^ s_tmp5.ui;

    // If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a rotation

    s_tmp5.f  = -2.f;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_tmp4.f  = 1.f;
    s_tmp4.f  = __fadd_rn(s_tmp4.f, s_tmp5.f);

    s_a12.f = s_a12.f * s_tmp4.f;
    s_a22.f = s_a22.f * s_tmp4.f;
    s_a32.f = s_a32.f * s_tmp4.f;

    s_v12.f = s_v12.f * s_tmp4.f;
    s_v22.f = s_v22.f * s_tmp4.f;
    s_v32.f = s_v32.f * s_tmp4.f;

    // Swap columns 1-3 if necessary

    s_tmp4.ui = (s_tmp1.f < s_tmp3.f) ? 0xffffffff : 0;
    s_tmp5.ui = s_a11.ui ^ s_a13.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a11.ui  = s_a11.ui ^ s_tmp5.ui;
    s_a13.ui  = s_a13.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_a21.ui ^ s_a23.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a21.ui  = s_a21.ui ^ s_tmp5.ui;
    s_a23.ui  = s_a23.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_a31.ui ^ s_a33.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a31.ui  = s_a31.ui ^ s_tmp5.ui;
    s_a33.ui  = s_a33.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v11.ui ^ s_v13.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v11.ui  = s_v11.ui ^ s_tmp5.ui;
    s_v13.ui  = s_v13.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v21.ui ^ s_v23.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v21.ui  = s_v21.ui ^ s_tmp5.ui;
    s_v23.ui  = s_v23.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v31.ui ^ s_v33.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v31.ui  = s_v31.ui ^ s_tmp5.ui;
    s_v33.ui  = s_v33.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_tmp1.ui ^ s_tmp3.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_tmp1.ui = s_tmp1.ui ^ s_tmp5.ui;
    s_tmp3.ui = s_tmp3.ui ^ s_tmp5.ui;

    // If columns 1-3 have been swapped, negate 1st column of A and V so that V is still a rotation

    s_tmp5.f  = -2.f;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_tmp4.f  = 1.f;
    s_tmp4.f  = __fadd_rn(s_tmp4.f, s_tmp5.f);

    s_a11.f = s_a11.f * s_tmp4.f;
    s_a21.f = s_a21.f * s_tmp4.f;
    s_a31.f = s_a31.f * s_tmp4.f;

    s_v11.f = s_v11.f * s_tmp4.f;
    s_v21.f = s_v21.f * s_tmp4.f;
    s_v31.f = s_v31.f * s_tmp4.f;

    // Swap columns 2-3 if necessary

    s_tmp4.ui = (s_tmp2.f < s_tmp3.f) ? 0xffffffff : 0;
    s_tmp5.ui = s_a12.ui ^ s_a13.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a12.ui  = s_a12.ui ^ s_tmp5.ui;
    s_a13.ui  = s_a13.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_a22.ui ^ s_a23.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a22.ui  = s_a22.ui ^ s_tmp5.ui;
    s_a23.ui  = s_a23.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_a32.ui ^ s_a33.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_a32.ui  = s_a32.ui ^ s_tmp5.ui;
    s_a33.ui  = s_a33.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v12.ui ^ s_v13.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v12.ui  = s_v12.ui ^ s_tmp5.ui;
    s_v13.ui  = s_v13.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v22.ui ^ s_v23.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v22.ui  = s_v22.ui ^ s_tmp5.ui;
    s_v23.ui  = s_v23.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_v32.ui ^ s_v33.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_v32.ui  = s_v32.ui ^ s_tmp5.ui;
    s_v33.ui  = s_v33.ui ^ s_tmp5.ui;

    s_tmp5.ui = s_tmp2.ui ^ s_tmp3.ui;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_tmp2.ui = s_tmp2.ui ^ s_tmp5.ui;
    s_tmp3.ui = s_tmp3.ui ^ s_tmp5.ui;

    // If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a rotation

    s_tmp5.f  = -2.f;
    s_tmp5.ui = s_tmp5.ui & s_tmp4.ui;
    s_tmp4.f  = 1.f;
    s_tmp4.f  = __fadd_rn(s_tmp4.f, s_tmp5.f);

    s_a13.f = s_a13.f * s_tmp4.f;
    s_a23.f = s_a23.f * s_tmp4.f;
    s_a33.f = s_a33.f * s_tmp4.f;

    s_v13.f = s_v13.f * s_tmp4.f;
    s_v23.f = s_v23.f * s_tmp4.f;
    s_v33.f = s_v33.f * s_tmp4.f;

    //###########################################################
    // Construct QR factorization of A*V (=U*D) using Givens rotations
    //###########################################################

    s_u11.f = 1.f;
    s_u12.f = 0.f;
    s_u13.f = 0.f;
    s_u21.f = 0.f;
    s_u22.f = 1.f;
    s_u23.f = 0.f;
    s_u31.f = 0.f;
    s_u32.f = 0.f;
    s_u33.f = 1.f;

    s_sh.f	= s_a21.f * s_a21.f;
    s_sh.ui = (s_sh.f >= gsmall_number) ? 0xffffffff : 0;
    s_sh.ui = s_sh.ui & s_a21.ui;

    s_tmp5.f  = 0.f;
    s_ch.f	  = __fsub_rn(s_tmp5.f, s_a11.f);
    s_ch.f	  = max(s_ch.f, s_a11.f);
    s_ch.f	  = max(s_ch.f, gsmall_number);
    s_tmp5.ui = (s_a11.f >= s_tmp5.f) ? 0xffffffff : 0;

    s_tmp1.f = s_ch.f * s_ch.f;
    s_tmp2.f = s_sh.f * s_sh.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = __frsqrt_rn(s_tmp2.f);

    s_tmp4.f = s_tmp1.f * 0.5f;
    s_tmp3.f = s_tmp1.f * s_tmp4.f;
    s_tmp3.f = s_tmp1.f * s_tmp3.f;
    s_tmp3.f = s_tmp2.f * s_tmp3.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp1.f = __fsub_rn(s_tmp1.f, s_tmp3.f);
    s_tmp1.f = s_tmp1.f * s_tmp2.f;

    s_ch.f = __fadd_rn(s_ch.f, s_tmp1.f);

    s_tmp1.ui = ~s_tmp5.ui & s_sh.ui;
    s_tmp2.ui = ~s_tmp5.ui & s_ch.ui;
    s_ch.ui	  = s_tmp5.ui & s_ch.ui;
    s_sh.ui	  = s_tmp5.ui & s_sh.ui;
    s_ch.ui	  = s_ch.ui | s_tmp1.ui;
    s_sh.ui	  = s_sh.ui | s_tmp2.ui;

    s_tmp1.f = s_ch.f * s_ch.f;
    s_tmp2.f = s_sh.f * s_sh.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = __frsqrt_rn(s_tmp2.f);

    s_tmp4.f = s_tmp1.f * 0.5f;
    s_tmp3.f = s_tmp1.f * s_tmp4.f;
    s_tmp3.f = s_tmp1.f * s_tmp3.f;
    s_tmp3.f = s_tmp2.f * s_tmp3.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp1.f = __fsub_rn(s_tmp1.f, s_tmp3.f);

    s_ch.f = s_ch.f * s_tmp1.f;
    s_sh.f = s_sh.f * s_tmp1.f;

    s_c.f = s_ch.f * s_ch.f;
    s_s.f = s_sh.f * s_sh.f;
    s_c.f = __fsub_rn(s_c.f, s_s.f);
    s_s.f = s_sh.f * s_ch.f;
    s_s.f = __fadd_rn(s_s.f, s_s.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    s_tmp1.f = s_s.f * s_a11.f;
    s_tmp2.f = s_s.f * s_a21.f;
    s_a11.f	 = s_c.f * s_a11.f;
    s_a21.f	 = s_c.f * s_a21.f;
    s_a11.f	 = __fadd_rn(s_a11.f, s_tmp2.f);
    s_a21.f	 = __fsub_rn(s_a21.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_a12.f;
    s_tmp2.f = s_s.f * s_a22.f;
    s_a12.f	 = s_c.f * s_a12.f;
    s_a22.f	 = s_c.f * s_a22.f;
    s_a12.f	 = __fadd_rn(s_a12.f, s_tmp2.f);
    s_a22.f	 = __fsub_rn(s_a22.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_a13.f;
    s_tmp2.f = s_s.f * s_a23.f;
    s_a13.f	 = s_c.f * s_a13.f;
    s_a23.f	 = s_c.f * s_a23.f;
    s_a13.f	 = __fadd_rn(s_a13.f, s_tmp2.f);
    s_a23.f	 = __fsub_rn(s_a23.f, s_tmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    s_tmp1.f = s_s.f * s_u11.f;
    s_tmp2.f = s_s.f * s_u12.f;
    s_u11.f	 = s_c.f * s_u11.f;
    s_u12.f	 = s_c.f * s_u12.f;
    s_u11.f	 = __fadd_rn(s_u11.f, s_tmp2.f);
    s_u12.f	 = __fsub_rn(s_u12.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_u21.f;
    s_tmp2.f = s_s.f * s_u22.f;
    s_u21.f	 = s_c.f * s_u21.f;
    s_u22.f	 = s_c.f * s_u22.f;
    s_u21.f	 = __fadd_rn(s_u21.f, s_tmp2.f);
    s_u22.f	 = __fsub_rn(s_u22.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_u31.f;
    s_tmp2.f = s_s.f * s_u32.f;
    s_u31.f	 = s_c.f * s_u31.f;
    s_u32.f	 = s_c.f * s_u32.f;
    s_u31.f	 = __fadd_rn(s_u31.f, s_tmp2.f);
    s_u32.f	 = __fsub_rn(s_u32.f, s_tmp1.f);

    // Second Givens rotation

    s_sh.f	= s_a31.f * s_a31.f;
    s_sh.ui = (s_sh.f >= gsmall_number) ? 0xffffffff : 0;
    s_sh.ui = s_sh.ui & s_a31.ui;

    s_tmp5.f  = 0.f;
    s_ch.f	  = __fsub_rn(s_tmp5.f, s_a11.f);
    s_ch.f	  = max(s_ch.f, s_a11.f);
    s_ch.f	  = max(s_ch.f, gsmall_number);
    s_tmp5.ui = (s_a11.f >= s_tmp5.f) ? 0xffffffff : 0;

    s_tmp1.f = s_ch.f * s_ch.f;
    s_tmp2.f = s_sh.f * s_sh.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = __frsqrt_rn(s_tmp2.f);

    s_tmp4.f = s_tmp1.f * 0.5;
    s_tmp3.f = s_tmp1.f * s_tmp4.f;
    s_tmp3.f = s_tmp1.f * s_tmp3.f;
    s_tmp3.f = s_tmp2.f * s_tmp3.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp1.f = __fsub_rn(s_tmp1.f, s_tmp3.f);
    s_tmp1.f = s_tmp1.f * s_tmp2.f;

    s_ch.f = __fadd_rn(s_ch.f, s_tmp1.f);

    s_tmp1.ui = ~s_tmp5.ui & s_sh.ui;
    s_tmp2.ui = ~s_tmp5.ui & s_ch.ui;
    s_ch.ui	  = s_tmp5.ui & s_ch.ui;
    s_sh.ui	  = s_tmp5.ui & s_sh.ui;
    s_ch.ui	  = s_ch.ui | s_tmp1.ui;
    s_sh.ui	  = s_sh.ui | s_tmp2.ui;

    s_tmp1.f = s_ch.f * s_ch.f;
    s_tmp2.f = s_sh.f * s_sh.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = __frsqrt_rn(s_tmp2.f);

    s_tmp4.f = s_tmp1.f * 0.5f;
    s_tmp3.f = s_tmp1.f * s_tmp4.f;
    s_tmp3.f = s_tmp1.f * s_tmp3.f;
    s_tmp3.f = s_tmp2.f * s_tmp3.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp1.f = __fsub_rn(s_tmp1.f, s_tmp3.f);

    s_ch.f = s_ch.f * s_tmp1.f;
    s_sh.f = s_sh.f * s_tmp1.f;

    s_c.f = s_ch.f * s_ch.f;
    s_s.f = s_sh.f * s_sh.f;
    s_c.f = __fsub_rn(s_c.f, s_s.f);
    s_s.f = s_sh.f * s_ch.f;
    s_s.f = __fadd_rn(s_s.f, s_s.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    s_tmp1.f = s_s.f * s_a11.f;
    s_tmp2.f = s_s.f * s_a31.f;
    s_a11.f	 = s_c.f * s_a11.f;
    s_a31.f	 = s_c.f * s_a31.f;
    s_a11.f	 = __fadd_rn(s_a11.f, s_tmp2.f);
    s_a31.f	 = __fsub_rn(s_a31.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_a12.f;
    s_tmp2.f = s_s.f * s_a32.f;
    s_a12.f	 = s_c.f * s_a12.f;
    s_a32.f	 = s_c.f * s_a32.f;
    s_a12.f	 = __fadd_rn(s_a12.f, s_tmp2.f);
    s_a32.f	 = __fsub_rn(s_a32.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_a13.f;
    s_tmp2.f = s_s.f * s_a33.f;
    s_a13.f	 = s_c.f * s_a13.f;
    s_a33.f	 = s_c.f * s_a33.f;
    s_a13.f	 = __fadd_rn(s_a13.f, s_tmp2.f);
    s_a33.f	 = __fsub_rn(s_a33.f, s_tmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    s_tmp1.f = s_s.f * s_u11.f;
    s_tmp2.f = s_s.f * s_u13.f;
    s_u11.f	 = s_c.f * s_u11.f;
    s_u13.f	 = s_c.f * s_u13.f;
    s_u11.f	 = __fadd_rn(s_u11.f, s_tmp2.f);
    s_u13.f	 = __fsub_rn(s_u13.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_u21.f;
    s_tmp2.f = s_s.f * s_u23.f;
    s_u21.f	 = s_c.f * s_u21.f;
    s_u23.f	 = s_c.f * s_u23.f;
    s_u21.f	 = __fadd_rn(s_u21.f, s_tmp2.f);
    s_u23.f	 = __fsub_rn(s_u23.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_u31.f;
    s_tmp2.f = s_s.f * s_u33.f;
    s_u31.f	 = s_c.f * s_u31.f;
    s_u33.f	 = s_c.f * s_u33.f;
    s_u31.f	 = __fadd_rn(s_u31.f, s_tmp2.f);
    s_u33.f	 = __fsub_rn(s_u33.f, s_tmp1.f);

    // Third Givens Rotation

    s_sh.f	= s_a32.f * s_a32.f;
    s_sh.ui = (s_sh.f >= gsmall_number) ? 0xffffffff : 0;
    s_sh.ui = s_sh.ui & s_a32.ui;

    s_tmp5.f  = 0.f;
    s_ch.f	  = __fsub_rn(s_tmp5.f, s_a22.f);
    s_ch.f	  = max(s_ch.f, s_a22.f);
    s_ch.f	  = max(s_ch.f, gsmall_number);
    s_tmp5.ui = (s_a22.f >= s_tmp5.f) ? 0xffffffff : 0;

    s_tmp1.f = s_ch.f * s_ch.f;
    s_tmp2.f = s_sh.f * s_sh.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = __frsqrt_rn(s_tmp2.f);

    s_tmp4.f = s_tmp1.f * 0.5f;
    s_tmp3.f = s_tmp1.f * s_tmp4.f;
    s_tmp3.f = s_tmp1.f * s_tmp3.f;
    s_tmp3.f = s_tmp2.f * s_tmp3.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp1.f = __fsub_rn(s_tmp1.f, s_tmp3.f);
    s_tmp1.f = s_tmp1.f * s_tmp2.f;

    s_ch.f = __fadd_rn(s_ch.f, s_tmp1.f);

    s_tmp1.ui = ~s_tmp5.ui & s_sh.ui;
    s_tmp2.ui = ~s_tmp5.ui & s_ch.ui;
    s_ch.ui	  = s_tmp5.ui & s_ch.ui;
    s_sh.ui	  = s_tmp5.ui & s_sh.ui;
    s_ch.ui	  = s_ch.ui | s_tmp1.ui;
    s_sh.ui	  = s_sh.ui | s_tmp2.ui;

    s_tmp1.f = s_ch.f * s_ch.f;
    s_tmp2.f = s_sh.f * s_sh.f;
    s_tmp2.f = __fadd_rn(s_tmp1.f, s_tmp2.f);
    s_tmp1.f = __frsqrt_rn(s_tmp2.f);

    s_tmp4.f = s_tmp1.f * 0.5f;
    s_tmp3.f = s_tmp1.f * s_tmp4.f;
    s_tmp3.f = s_tmp1.f * s_tmp3.f;
    s_tmp3.f = s_tmp2.f * s_tmp3.f;
    s_tmp1.f = __fadd_rn(s_tmp1.f, s_tmp4.f);
    s_tmp1.f = __fsub_rn(s_tmp1.f, s_tmp3.f);

    s_ch.f = s_ch.f * s_tmp1.f;
    s_sh.f = s_sh.f * s_tmp1.f;

    s_c.f = s_ch.f * s_ch.f;
    s_s.f = s_sh.f * s_sh.f;
    s_c.f = __fsub_rn(s_c.f, s_s.f);
    s_s.f = s_sh.f * s_ch.f;
    s_s.f = __fadd_rn(s_s.f, s_s.f);

    //###########################################################
    // Rotate matrix A
    //###########################################################

    s_tmp1.f = s_s.f * s_a21.f;
    s_tmp2.f = s_s.f * s_a31.f;
    s_a21.f	 = s_c.f * s_a21.f;
    s_a31.f	 = s_c.f * s_a31.f;
    s_a21.f	 = __fadd_rn(s_a21.f, s_tmp2.f);
    s_a31.f	 = __fsub_rn(s_a31.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_a22.f;
    s_tmp2.f = s_s.f * s_a32.f;
    s_a22.f	 = s_c.f * s_a22.f;
    s_a32.f	 = s_c.f * s_a32.f;
    s_a22.f	 = __fadd_rn(s_a22.f, s_tmp2.f);
    s_a32.f	 = __fsub_rn(s_a32.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_a23.f;
    s_tmp2.f = s_s.f * s_a33.f;
    s_a23.f	 = s_c.f * s_a23.f;
    s_a33.f	 = s_c.f * s_a33.f;
    s_a23.f	 = __fadd_rn(s_a23.f, s_tmp2.f);
    s_a33.f	 = __fsub_rn(s_a33.f, s_tmp1.f);

    //###########################################################
    // Update matrix U
    //###########################################################

    s_tmp1.f = s_s.f * s_u12.f;
    s_tmp2.f = s_s.f * s_u13.f;
    s_u12.f	 = s_c.f * s_u12.f;
    s_u13.f	 = s_c.f * s_u13.f;
    s_u12.f	 = __fadd_rn(s_u12.f, s_tmp2.f);
    s_u13.f	 = __fsub_rn(s_u13.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_u22.f;
    s_tmp2.f = s_s.f * s_u23.f;
    s_u22.f	 = s_c.f * s_u22.f;
    s_u23.f	 = s_c.f * s_u23.f;
    s_u22.f	 = __fadd_rn(s_u22.f, s_tmp2.f);
    s_u23.f	 = __fsub_rn(s_u23.f, s_tmp1.f);

    s_tmp1.f = s_s.f * s_u32.f;
    s_tmp2.f = s_s.f * s_u33.f;
    s_u32.f	 = s_c.f * s_u32.f;
    s_u33.f	 = s_c.f * s_u33.f;
    s_u32.f	 = __fadd_rn(s_u32.f, s_tmp2.f);
    s_u33.f	 = __fsub_rn(s_u33.f, s_tmp1.f);

    v11 = s_v11.f;
    v12 = s_v12.f;
    v13 = s_v13.f;
    v21 = s_v21.f;
    v22 = s_v22.f;
    v23 = s_v23.f;
    v31 = s_v31.f;
    v32 = s_v32.f;
    v33 = s_v33.f;

    u11 = s_u11.f;
    u12 = s_u12.f;
    u13 = s_u13.f;
    u21 = s_u21.f;
    u22 = s_u22.f;
    u23 = s_u23.f;
    u31 = s_u31.f;
    u32 = s_u32.f;
    u33 = s_u33.f;

    s11 = s_a11.f;
    //s12 = s_a12.f; s13 = s_a13.f; s21 = s_a21.f;
    s22 = s_a22.f;
    //s23 = s_a23.f; s31 = s_a31.f; s32 = s_a32.f;
    s33 = s_a33.f;
}









#endif // HELPER_MATH_CUH
