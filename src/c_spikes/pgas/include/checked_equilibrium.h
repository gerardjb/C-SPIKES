#ifndef CSP_CHECKED_EQUILIBRIUM_H
#define CSP_CHECKED_EQUILIBRIUM_H
#include <array>
#include <limits>
#include <memory>
#include <optional>
class GCaMP;
namespace checked {
namespace detail {
using Matrix9=std::array<double,81>;
using Vector9=std::array<double,9>;
struct Factors { Matrix9 u{},v{}; Vector9 s{}; int code=-1; };
enum class Outcome { input_invalid,backend_failed,malformed_factors,svd_quality_failed,
 no_nullspace,nonunique,rank_ambiguous,normalization_unsafe,analytic_no_selection,invariant_failed,candidate };
struct Result {
 Outcome outcome=Outcome::input_invalid;
 bool backend_called=false,has_diagnostics=false,has_state=false,sorted=false;
 Factors factors; Matrix9 projector{};Vector9 state{};
 std::array<int,3> ranks{};int nullity=-1;
 double tau=0,reconstruction=0,u_orthogonality=0,v_orthogonality=0;
 double sum=std::numeric_limits<double>::quiet_NaN(),guard=std::numeric_limits<double>::quiet_NaN();
 double residual=std::numeric_limits<double>::quiet_NaN(),mass_error=std::numeric_limits<double>::quiet_NaN(),minimum=std::numeric_limits<double>::quiet_NaN();
};
using Backend=Factors (*)(const Matrix9&);
Factors gsl_backend(const Matrix9&);
std::array<int,3> classify(const Vector9&);
bool normalize(const Vector9&,double,Vector9&,double&,double&);
bool positive(const Vector9&,double);
Result assess(const Matrix9&,const Factors&,bool,double);
Result evaluate(const Matrix9&,bool,double,Backend=gsl_backend);
const char* outcome_name(Outcome);
const char* normalization_name(Outcome);
}
enum class Status { success, input_invalid, backend_failed, malformed_factors,
 svd_quality_failed, no_nullspace, nonunique, rank_ambiguous, normalization_unsafe,
 analytic_no_selection, invariant_failed, context_required, context_busy };
enum class Operation { create, parameters, state, initialize, parameters_and_init, equilibrium };
struct CellParameters { double G_tot,gamma,DCaT,Rf,gam_in,gam_out; };
class Context {
 bool owned_=false;
 explicit Context(bool owned):owned_(owned){}
public:
 Context()=default;
 // A declaration by the host, not a detector of other libraries or threads.
 static Context owned_isolated_process(){return Context(true);}
 bool declared_exclusive() const{return owned_;}
};
struct CallbackDiagnostic {
 int count=0,code=0,line=0; bool truncated=false;
 std::array<char,128> reason{},file{};
};
struct Report {
 Status status=Status::input_invalid; Operation operation=Operation::equilibrium;
 bool committed=false; int solve_count=0,backend_code=0;
 CallbackDiagnostic callback;
 std::array<detail::Result,4> solves{};
};
struct UpdateResult { Report report; };
struct EquilibriumResult { Report report; std::optional<std::array<double,12>> state; };
struct CreateResult { Report report; std::unique_ptr<GCaMP> model; };
CreateResult try_create_gcamp(const std::array<double,14>&,CellParameters,const Context&);
const char* status_name(Status);
struct Engine;
#ifdef C_SPIKES_CHECKED_TESTING
struct TestAccess;
namespace testing {
struct Fault { int call=0;Status status=Status::backend_failed;int callback_mode=0;bool unwind=false;void(*nested)()=nullptr; };
Fault& fault();
void reset();
}
#endif
}
#endif
