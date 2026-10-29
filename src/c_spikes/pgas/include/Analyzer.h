#ifndef ANALYZER_H
#define ANALYZER_H

#include <string>
#include "application_observation.h"
#include "history_storage.h"

class Analyzer {
public:

    Analyzer(const csp_application::ObservationVector& time,const csp_application::ObservationVector& data, const std::string& constants_file, const std::string& output_folder,
                 unsigned int column, const std::string& tag, unsigned int niter, const std::string& trainedPriorFile,
                 bool append, unsigned int trim, bool verbose, const csp_application::ObservationVector& gtSpikes,
                 bool has_trained_priors = false, bool has_gtspikes = false, unsigned int maxlen = 0, const std::string& Gparam_file = "",
                 int seed=0);

    void run();
    //These are the methods for dealing with the pgas time-independent parameters
    void add_parameter_sample(std::vector<double> parameter_sample);
    const csp_model::HistoryMatrix& get_parameter_samples() const {
        return parameter_samples;
    }
	std::vector<double> final_params;

private:
    csp_application::ObservationVector time;
    csp_application::ObservationVector data;
    std::string constants_file;
    std::string output_folder;
    unsigned int column;
    csp_application::ObservationVector gtSpikes;
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

	csp_model::HistoryMatrix parameter_samples;
};

#endif // ANALYZER_H
