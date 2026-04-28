#ifndef ANALYZER_H
#define ANALYZER_H

#include <string>

class Analyzer {
public:

    Analyzer(const arma::vec& time,const arma::vec& data, const std::string& constants_file, const std::string& output_folder,
                 unsigned int column, const std::string& tag, unsigned int niter = 0, const std::string& trainedPriorFile = "",
                 bool append = false, unsigned int trim = 1, bool verbose = true, const arma::vec& gtSpikes=0,
                 bool has_trained_priors = false, bool has_gtspikes = false, unsigned int maxlen = 0, const std::string& Gparam_file = "",
                 int seed=0);

    void run();
    //These are the methods for dealing with the pgas time-independent parameters
    void add_parameter_sample(std::vector<double> parameter_sample);
    const arma::mat& get_parameter_samples() const {
        return parameter_samples;
    }
	std::vector<double> final_params;

private:
    arma::vec time;
    arma::vec data;
    std::string constants_file;
    std::string output_folder;
    unsigned int column;
    arma::vec gtSpikes;
    std::string tag;
    int seed;

    std::string trainedPriorFile;
    bool has_trained_priors;
    bool has_gtspikes;
    std::string Gparam_file;
    bool append;
    unsigned int niter;
    unsigned int trim;
    bool verbose;
    unsigned int maxlen;

	arma::mat parameter_samples;	
};

#endif // ANALYZER_H
