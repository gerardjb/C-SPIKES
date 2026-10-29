#ifndef I2B_OBSERVER_H
#define I2B_OBSERVER_H
#include <vector>
#include <fstream>
#include <stdexcept>
namespace i2b {
template<class V> std::vector<double> copy(const V& v){std::vector<double> x; x.reserve(v.size());for(size_t i=0;i<v.size();++i)x.push_back(v(i));return x;}
inline std::vector<double> read_params(const char* p){std::ifstream f(p);if(!f)throw std::runtime_error("fixtures");std::vector<double> x;double v;while(f>>v)x.push_back(v);if(!f.eof())throw std::runtime_error("fixtures");return x;}
}
namespace l3e {
struct Snapshot {int calls=0,M=0;double fine_dt=0,new_t_end=0;std::vector<double> time,grid,states,spikes,input;};
inline Snapshot& snapshot(){static thread_local Snapshot s;return s;}
template<class T,class V,class Mx> void capture(int M,double dt,double end,const T& time,const V& grid,const Mx& G,const V& B,const V& C,const V& I,const T& spikes,const V& input){auto& s=snapshot();++s.calls;s.M=M;s.fine_dt=dt;s.new_t_end=end;if(grid.size()>64||time.size()>64||G.rows()!=grid.size()||G.cols()!=9||B.size()!=grid.size()||C.size()!=grid.size()||I.size()!=grid.size())throw std::runtime_error("observer shape/cap");s.time=i2b::copy(time);s.grid=i2b::copy(grid);s.spikes=i2b::copy(spikes);s.input=i2b::copy(input);s.states.clear();s.states.reserve(grid.size()*12);for(size_t r=0;r<grid.size();++r){for(size_t c=0;c<9;++c)s.states.push_back(G(r,c));s.states.push_back(B(r));s.states.push_back(C(r));s.states.push_back(I(r));}}
}
namespace l3f {
struct Snapshot {std::vector<double> grid,input,states,tail;};inline Snapshot& snapshot(){static thread_local Snapshot s;return s;}
template<class V,class Mx> void capture(const V& grid,const V& input,const Mx& G,const V& B,const V& C,const V& I,const V& F){auto& s=snapshot();s=Snapshot{};if(grid.size()>64||G.rows()!=grid.size()||G.cols()!=9)throw std::runtime_error("observer cap/shape");s.grid=i2b::copy(grid);s.input=i2b::copy(input);for(size_t r=0;r<grid.size();++r){for(size_t c=0;c<9;++c)s.states.push_back(G(r,c));s.states.push_back(B(r));s.states.push_back(C(r));s.states.push_back(I(r));}for(size_t i=1;i<F.size();++i)s.tail.push_back(F(i));}
}
#endif
