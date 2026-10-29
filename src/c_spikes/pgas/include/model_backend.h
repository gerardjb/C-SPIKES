#ifndef CSP_MODEL_BACKEND_H
#define CSP_MODEL_BACKEND_H
#include "model_workspace.h"
#include "generator9.h"
namespace csp_model { namespace workspace {
// Private qualified CPU backend; retains the existing owned generator snapshot.
void product9_compat(const Generator9&,const Indicator9&,Indicator9&);
}}
#endif
