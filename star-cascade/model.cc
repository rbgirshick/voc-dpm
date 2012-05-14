#include "mex.h"
#include "model.h"

// see: model.h for descriptions of each class field.

// handy accessors

// return field from struct a
static inline const mxArray *F(const mxArray *a, const char *field) {
  return mxGetField(a, 0, field);
}

// return pointer to field from struct a
template <typename T>
static inline T *Fpr(const mxArray *a, const char *field) {
  return mxGetPr(F(a, field));
}

// return scalar of field from struct a
template <typename T>
static inline T Fsc(const mxArray *a, const char *field) {
  return mxGetScalar(F(a, field));
}

// return field from struct in cell of struct array a
static inline const mxArray *CF(const mxArray *a, int cell, const char *field) {
  return F(mxGetCell(a, cell), field);
}

void Model::initmodel(const mxArray *model) {
  thresh        = Fsc<double>(model, "thresh");
  interval      = (int)Fsc<double>(model, "interval");
  numcomponents = (int)Fsc<double>(model, "numcomponents");
  sbin          = (int)Fsc<double>(model, "sbin");
  
  const mxArray *components = F(model, "components");
  const mxArray *definfos   = F(model, "defs");
  const mxArray *partinfos  = F(model, "partfilters");
  const mxArray *rootinfos  = F(model, "rootfilters");
  numpartfilters            = (int)(mxGetDimensions(partinfos)[1]);
  numdefparams              = (int)(mxGetDimensions(definfos)[1]);

  numparts        = new int[numcomponents];
  anchors         = new double**[numcomponents];
  defs            = new double*[numdefparams];
  rootfilters     = new float*[numcomponents];
  partfilters[0]  = new float*[numpartfilters];
  partfilters[1]  = new float*[numpartfilters];
  rootfilterdims  = new mwSize*[numcomponents];
  partfilterdims  = new mwSize*[numpartfilters];
  pfind           = new int*[numcomponents];
  defind          = new int*[numcomponents];
  loc_scores      = new double*[numcomponents];
  for (int i = 0; i < numpartfilters; i++) {
    const mxArray *partinfo = mxGetCell(partinfos, i);
    const mxArray *w        = F(partinfo, "w");
    partfilters[0][i]       = (float *)mxGetPr(w);
    partfilterdims[i]       = (mwSize*)mxGetDimensions(w);
    w                       = F(partinfo, "wpca");
    partfilters[1][i]       = (float *)mxGetPr(w);
  }
  for (int i = 0; i < numdefparams; i++) {
    const mxArray *definfo  = mxGetCell(definfos, i);
    defs[i]                 = Fpr<double>(definfo, "w");
  }
  const mxArray *cascadeinfo = F(model, "cascade");
  const mxArray *orderinfo   = F(cascadeinfo, "order");
  const mxArray *mxt         = F(cascadeinfo, "t");
  partorder                  = new int*[numcomponents];
  offsets                    = new double[numcomponents];
  t                          = new double*[numcomponents];
  for (int i = 0; i < numcomponents; i++) {
    const mxArray *parts  = CF(components, i, "parts");
    const mxArray *w      = CF(rootinfos, i, "w");
    numparts[i]           = mxGetDimensions(parts)[1];
    rootfilters[i]        = (float *)mxGetPr(w);
    rootfilterdims[i]     = (mwSize*)mxGetDimensions(w);
    anchors[i]            = new double*[numparts[i]];
    pfind[i]              = new int[numparts[i]];
    defind[i]             = new int[numparts[i]];
    offsets[i]            = mxGetScalar(mxGetField(mxGetCell(mxGetField(model, 0, "offsets"), i), 0, "w"));
    partorder[i]          = new int[2*numparts[i]+2];
    double *ord           = mxGetPr(mxGetCell(orderinfo, i));
    t[i]                  = mxGetPr(mxGetCell(mxt, i));

    for (int j = 0; j < numparts[i]; j++) {
      int dind                = (int)mxGetScalar(CF(parts, j, "defindex")) - 1;
      int pind                = (int)mxGetScalar(CF(parts, j, "partindex")) - 1;
      const mxArray *definfo  = mxGetCell(definfos, dind);
      anchors[i][j]           = Fpr<double>(definfo, "anchor");
      pfind[i][j]             = pind;
      defind[i][j]            = dind;
    }
    // subtract 1 so that non-root parts are zero-indexed
    for (int j = 0; j < 2*numparts[i]+2; j++)
      partorder[i][j] = (int)ord[j] - 1;
  }
}

void Model::initpyramid(const mxArray *pyramid, const mxArray *projpyramid) {
  mxArray *mx_feat = mxGetField(pyramid, 0, "feat");
  mxArray *mx_proj_feat = mxGetField(projpyramid, 0, "feat");
  numlevels    = (int)(mxGetDimensions(mx_feat)[0]);
  featdims     = new int*[numlevels];
  featdimsprod = new int[numlevels];
  feat[0]      = new float*[numlevels];
  feat[1]      = new float*[numlevels];
  for (int l = 0; l < numlevels; l++) {
    const mxArray *mxA  = mxGetCell(mx_feat, l);
    featdims[l]         = (int*)mxGetDimensions(mxA);
    featdimsprod[l]     = featdims[l][0]*featdims[l][1];
    feat[0][l]          = (float *)mxGetPr(mxA);
    // projected pyramid
    mxA                 = mxGetCell(mx_proj_feat, l);
    feat[1][l]          = (float *)mxGetPr(mxA);
  }
  // Location/scale scores
  for (int c = 0; c < numcomponents; c++)
    loc_scores[c] = mxGetPr(mxGetCell(F(pyramid, "loc_scores"), c));
  numfeatures = mxGetDimensions(mxGetCell(mx_feat, 0))[2];
  pcadim = mxGetDimensions(mxGetCell(mx_proj_feat, 0))[2];
}

Model::~Model() {
  for (int i = 0; i < numcomponents; i++) {
    delete [] partorder[i];
    delete [] anchors[i];
    delete [] defind[i];
    delete [] pfind[i];
  }
  delete [] loc_scores;
  delete [] partorder;
  delete [] t;
  delete [] numparts;
  delete [] offsets;
  delete [] defind;
  delete [] pfind;
  delete [] anchors;
  delete [] defs;
  delete [] rootfilters;
  delete [] rootfilterdims;
  delete [] partfilters[0];
  delete [] partfilters[1];
  delete [] partfilterdims;
  delete [] featdims;
  delete [] featdimsprod;
  delete [] feat[0];
  delete [] feat[1];
}
