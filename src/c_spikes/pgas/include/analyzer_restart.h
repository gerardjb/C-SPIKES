#ifndef C_SPIKES_ANALYZER_RESTART_H
#define C_SPIKES_ANALYZER_RESTART_H
#include <array>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

// Current output schema only. This is a parameter restart, not an RNG/trajectory checkpoint.
namespace analyzer_restart {
struct Record { std::array<double,11> values{}; std::uint64_t rows=0; };
inline bool numbers(const std::string& line, std::size_t count, std::vector<double>& values) {
    values.clear(); std::istringstream stream(line); std::string token;
    while (std::getline(stream,token,',')) {
        try {
            std::size_t end=0; double value=std::stod(token,&end);
            if (token.find_first_not_of(" \t\r",end)!=std::string::npos) return false;
            values.push_back(value);
        } catch (const std::invalid_argument&) { return false; }
          catch (const std::out_of_range&) { return false; }
    }
    return !line.empty() && line.back()!=',' && values.size()==count;
}
inline bool read(const std::string& root,const std::string& tag,std::size_t time_count,
                 Record& record,std::string& message) {
    std::ifstream params(root+"/param_samples_"+tag+".dat"); std::string line;
    if (!std::getline(params,line) || line!="G_tot,gamma,DCaT,Rf,gam_in,gam_out,sigma2,r0,r1,w01,w10") {
        message="missing or unsupported parameter header (eleven-column schema required)"; return false;
    }
    std::vector<double> values;
    while (std::getline(params,line)) {
        if (!numbers(line,11,values)) { message="malformed parameter row"; return false; }
        std::copy(values.begin(),values.end(),record.values.begin()); ++record.rows;
    }
    if (params.bad() || record.rows==0) { message="empty or unreadable parameter history"; return false; }
    if (time_count==0 || record.rows>std::numeric_limits<std::uint64_t>::max()/time_count) {
        message="invalid trajectory record dimensions"; return false;
    }
    std::ifstream trajectories(root+"/traj_samples_"+tag+".dat");
    if (!std::getline(trajectories,line) || line!="index,burst,B,S,C,Y") {
        message="missing or unsupported trajectory header"; return false;
    }
    std::uint64_t count=0;
    while (std::getline(trajectories,line)) {
        if (!numbers(line,6,values) || !std::isfinite(values[0]) ||
            values[0]!=static_cast<double>(count/time_count)) {
            message="malformed or nonconsecutive trajectory record"; return false;
        }
        ++count;
    }
    if (trajectories.bad() || count!=record.rows*time_count) {
        message="trajectory/parameter record counts differ"; return false;
    }
    std::ifstream likelihood(root+"/logp_"+tag+".dat"); count=0;
    while (std::getline(likelihood,line)) {
        if (!numbers(line,1,values)) { message="malformed logp row"; return false; }
        ++count;
    }
    if (likelihood.bad() || count!=record.rows) {
        message="logp/parameter record counts differ"; return false;
    }
    return true;
}
} // namespace analyzer_restart
#endif
