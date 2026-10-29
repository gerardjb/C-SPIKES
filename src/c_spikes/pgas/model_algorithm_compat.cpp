// SPDX-License-Identifier: Apache-2.0
// Portions adapted from Armadillo 14.2.2: fn_interp1.hpp, op_find_unique_meat.hpp,
// op_sort_index_meat.hpp, op_max_meat.hpp, diskio_meat.hpp and Mat_meat.hpp.
// Copyright 2008-2024 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Modified for C-SPIKES: fixed double-only owned storage and narrow private APIs.
// Licensed under the Apache License, Version 2.0; see tests/native/g2ah/LICENSE.armadillo.
// Provided AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#include "include/model_algorithm_compat.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace csp_model {
namespace {
struct Packet { double val;std::size_t index; };
struct Less { bool operator()(const Packet& a,const Packet& b) const { return a.val<b.val; } };
std::vector<std::size_t> unique_indices(ScalarInputView x) {
    std::vector<std::size_t> indices(x.size());
    std::vector<Packet> packets(x.size());
    for(std::size_t i=0;i<x.size();++i){if(std::isnan(x(i)))throw std::logic_error("find_unique(): detected NaN");packets[i]={x(i),i};}
    std::sort(packets.begin(),packets.end(),Less{});
    indices[0]=packets[0].index;std::size_t count=1;
    for(std::size_t i=1;i<x.size();++i){const double diff=packets[i-1].val-packets[i].val;if(diff!=0.)indices[count++]=packets[i].index;}
    indices.resize(count);return indices;
}
bool token_value(double& value,const std::string& token) {
    const auto n=token.length();const char* s=token.c_str();
    if(n==0||(n==1&&s[0]=='0')){value=0.;return true;}
    if(n==3||n==4){bool neg=s[0]=='-',pos=s[0]=='+';std::size_t off=((neg||pos)&&n==4)?1:0;
        char a=s[off],b=s[off+1],c=s[off+2];
        if((a=='i'||a=='I')&&(b=='n'||b=='N')&&(c=='f'||c=='F')){value=neg?-std::numeric_limits<double>::infinity():std::numeric_limits<double>::infinity();return true;}
        if((a=='n'||a=='N')&&(b=='a'||b=='A')&&(c=='n'||c=='N')){value=std::numeric_limits<double>::quiet_NaN();return true;}
    }
    char* end=nullptr;value=std::strtod(s,&end);return end!=s;
}
}
HistoryVector linear_interpolate_compat(ScalarInputView x,ScalarInputView y,ScalarInputView query) {
    if(x.size()!=y.size())throw std::logic_error("interp1(): X and Y must have the same number of elements");
    if(x.size()<2)throw std::logic_error("interp1(): X must have at least two unique elements");
    std::vector<std::size_t> indices;
    try { indices=unique_indices(x); } catch(...) { }
    const auto n=indices.size();
    if(n<2)throw std::logic_error("interp1(): X must have at least two unique elements");
    std::vector<double> xs(n),ys(n);
    for(std::size_t i=0;i<n;++i){xs[i]=x(indices[i]);ys[i]=y(indices[i]);}
    bool sorted=true;
    for(std::size_t i=1;i<query.size();++i)if(query(i-1)>query(i)){sorted=false;break;}
    std::vector<Packet> packets;std::vector<double> qs;
    if(!sorted){
        packets.resize(query.size());
        for(std::size_t i=0;i<query.size();++i){if(std::isnan(query(i)))throw std::logic_error("sort_index(): detected NaN");packets[i]={query(i),i};}
        std::sort(packets.begin(),packets.end(),Less{});qs.resize(query.size());
        for(std::size_t i=0;i<query.size();++i)qs[i]=query(packets[i].index);
    }
    // Bounds are only used for comparisons; sorted support contains no NaNs.
    const double minimum=xs.front(),maximum=xs.back();
    HistoryVector out;out.set_size(query.size());std::size_t a=0,b=0;
    const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    for(std::size_t i=0;i<query.size();++i){
        const double q=sorted?query(i):qs[i];
        if(q<minimum||q>maximum)out(i)=nan;
        else if(std::isnan(q))out(i)=nan;
        else{
            double ae=inf,be=inf;
            for(std::size_t j=a;j<n;++j){const double tmp=xs[j]-q;const double err=tmp>=0.?tmp:-tmp;if(err>=ae)break;else{ae=err;a=j;}}
            if(xs[a]-q<=0.)b=a+1<n?a+1:a;else b=a>=1?a-1:a;
            be=std::abs(xs[b]-q);
            if(a>b){std::swap(a,b);std::swap(ae,be);}
            const double weight=ae>0.?ae/(ae+be):0.;
            out(i)=(1.-weight)*ys[a]+weight*ys[b];
        }
    }
    if(!sorted&&out.size()){HistoryVector restored;restored.set_size(out.size());for(std::size_t i=0;i<out.size();++i)restored(packets[i].index)=out(i);out=std::move(restored);}
    return out;
}
double max100_compat(const std::array<double,100>& values) {
    double a=-std::numeric_limits<double>::infinity(),b=a;
    for(std::size_t i=0,j=1;j<100;i+=2,j+=2){const double x=values[i],y=values[j];if(x>a)a=x;if(y>b)b=y;}
    return a>b?a:b;
}
namespace detail {
bool load_raw_ascii_values(double* destination,std::size_t size,const std::string& filename) {
    // The pinned Linux configuration uses binary text streams; locale is unchanged.
    std::ifstream f(filename,std::ios::binary);bool ok=f.is_open();
    if(ok){
        ok=f.good();f.clear();const std::fstream::pos_type start=f.tellg();
        std::size_t rows=0,cols=0;bool found=false;std::string line,token;std::stringstream stream;
        while(f.good()&&ok){std::getline(f,line);if(line.size()==0)break;stream.clear();stream.str(line);std::size_t count=0;while(stream>>token)++count;if(!found){cols=count;found=true;}else if(count!=cols)ok=false;++rows;}
        if(ok){f.clear();f.seekg(start);if(f.fail()||f.tellg()!=start)ok=false;
            else if(rows!=size||cols!=1)ok=false;
            else for(std::size_t row=0;row<size&&ok;++row){f>>token;ok=token_value(destination[row],token);}
        }
        f.close();
    }
    if(!ok)for(std::size_t i=0;i<size;++i)destination[i]=0.;
    return ok;
}
}
}
