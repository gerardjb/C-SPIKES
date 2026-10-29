#ifndef CSP_MODEL_EVOLUTION_COMPAT_H
#define CSP_MODEL_EVOLUTION_COMPAT_H
#include "model_workspace.h"
#include "scalar_input_view.h"
namespace csp_model { namespace workspace {
HistoryVector gaussian_input_compat(ScalarInputView,double,double,double);
HistoryVector zeroed_temporary_compat(std::size_t);
void copy_indicator_row_compat(HistoryMatrix&,std::size_t,const Indicator9&);
#ifdef CSP_G2AJ_TEST_OBSERVER
int gaussian_test_parallel();
int gaussian_test_team();
#endif
}}
#endif
