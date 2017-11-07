#include "TH.h"
#include "THNN.h"

inline real log_add(real x, real y) {
  if (fabs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline real log_mul(real x, real y) { return x + y; }

static void forward_algorithm(THtTensor &lr, THTensor &lmatch, real skip = -5) {
  int n = THTensor(_size)(lmatch, 0);
  int m = THTensor(_size)(lmatch, 1);
  THTensor_(resize2d)(lr, n, m)
  THTensor_(zero)(lr);
  THTensor *v = THTensor_(newWithSize1d)(m);
  THTensor *w = THTensor_(newWithSize1d)(m);
  for (int j = 0; j < m; j++) {
    THTensor_(set1d)(v, j, skip * j);
  }
  for (int i = 0; i < n; i++) {
    THTensor_(set1d)(w, 0, skip * i);
    for (int j = 1; j < m; j++) {
      THTensor_(set1d)(w, j, THTensor_(get1d(v, j-1));
    }
    for (int j = 0; j < m; j++) {
      real same = log_mul(THTensor_(get1d)(v, j), THTensor_(get2d)(lmatch, i, j));
      real next = log_mul(THTensor_(get1d)(w, j), THTensor_(get2d)(lmatch, i, j));
      THTensor_(set1d)(v, j, log_add(same, next));
    }
    for (int j = 0; j < m; j++) {
      THTensor_(set2d)(lr, i, j, THTensor_(get1d(v, j));
    }
  }
}

static void forwardbackward(THTensor &both, THTensor &lmatch) {
  int n = THTensor_(size)(lmatch, 0);
  int m = THTensor_(size)(lmatch, 1);
  THTensor *lr = THTensor_(new)();
  forward_algorithm(lr, lmatch);
  THTensor *rlmatch = THTensor_(newWithSize2d)(n, m);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      THTensor_(set2d)(rlmatch, i, j, THTensor_(get2d)(lmatch, n-i-1, m-j-1));
    }
  }
  THTensor *rrl = THTensor_(new)();
  forward_algorithm(rrl, rlmatch);
  THTensor *rl = THTensor_(newWithSize2d)(n, m);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      THTensor_(set2d)(rl, i, j, THTensor_(get2d)(rrl, n-i-1, m-j-1));
    }
  }
  both = THTensor_(add)(lr, rl, 1.0);
}

void ctc_align_targets(THTensor &posteriors, THTensor &outputs,
                       THTensor &targets) {
  real lo = 1e-6;

  int n1 = THTensor_(size)(outputs, 0);
  int n2 = THTensor_(size)(targets, 0);
  int nc = THTensor_(size)(targets, 1);

  // compute log probability of state matches
  THTensor *lmatch = THTensor_(newWithSize2d)(n1, n2);
  THTensor_(zero)(lmatch);
  for (int t1 = 0; t1 < n1; t1++) {
    THTensor *out = THTensor_(newWithSize1d)(nc);
    for (int i = 0; i < nc; i++) {
      THTensor_(set1d)(out, i, fmax(lo, THTensor_(get2d)(outputs, t1, i)));
    }
    THTensor_(div)(out, THTensor_(sumall)(out));
    //out = out / out.sum();
    for (int t2 = 0; t2 < n2; t2++) {
      real total = 0.0;
      for (int k = 0; k < nc; k++) {
        total += THTensor_(get1d)(out, k) * THTensor_(get2d)(targets, t2, k)
      }
      THTensor_(set2d)(lmatch, t1, t2, log(total));
    }
  }
  // compute unnormalized forward backward algorithm
  THTensor *both;
  forwardbackward(both, lmatch);

  // compute normalized state probabilities
  THTensor *epath = THTensor_(new)();
  THTensor_(add)(epath, both, THTensor_(maxall)(both));
  // XXX: TH_TENSOR_APPLY?
  for(int i=0; i < THTensor_(size)(epath, 0); i++) {
      for(int j=0; j < THTensor_(size)(epath, 1); j++) {
          THTensor_(set2d)(epath, i, j, exp(THTensor_(get2d)(epath, i, j)))
      }
  }
  for (int j = 0; j < n2; j++) {
    real total = 0.0;
    for (int i = 0; i < THTensor_(size)(epath, 0); i++) {
      total += THTensor_(get2d)(epath, i, j);
    }
    total = fmax(1e-9, total);
    for (int i = 0; i < THTensor_(size)(epath, 0); i++) {
      THTensor_(set2d)(aligned, i, j, THTensor_(get2d)(aligned, i, j) / total);
    }
  }

  // compute posterior probabilities for each class and normalize
  THTensor *aligned = THTensor_(newWithSize2d)(n1, nc);
  THTensor_(zero)(aligned);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < nc; j++) {
      real total = 0.0;
      for (int k = 0; k < n2; k++) {
        real value = THTensor_(get2d)(epath, i, k) * THTensor_(get2d)(targets, k, j);
        total += value;
      }
      THTensor_(set2d)(aligned, i, j, total);
    }
  }
  for (int i = 0; i < n1; i++) {
    real total = 0.0;
    for (int j = 0; j < nc; j++) {
      total += THTensor_(get2d)(aligned, i, j);
    }
    total = fmax(total, 1e-9);a
    // XXX: should be vectorizable
    for (int j = 0; j < nc; j++) {
      THTensor_(set2d)(aligned, i, j, THTensor_(get2d)(aligned, i, j) / total);
    }
  }
  THTensor_(copy)(posteriors, aligned);
}

void THNN_(CTCCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output)
{
    THNN_ARGCHECK(input->nDimension == 3, 2, input,
                  "3D tensor expected for input, but got %s");
    THNN_CHECK_NELEMENT(input, target);

    THTensor_(resizeAs)(output, input);
    for(int i = 0; i < THTensor_(size)(input, 0); i++) {
        THTensor *out = THTensor_(newSelect)(output, 0, i);
        THTensor *in = THTensor_(newSelect)(input, 0, i);
        THTensor *t = THTensor_(newSelect)(targets, 0, i);
        ctc_align_targets(out, in, t);
    }
}

void THNN_(CTCCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
{
}
