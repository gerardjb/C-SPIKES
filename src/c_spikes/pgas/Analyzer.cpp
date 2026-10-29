#include "include/application_arithmetic_compat.h"
#include "include/application_result_compat.h"
#include "include/numerical_diagnostics.h"
#include <iostream>
#include <iomanip>
#include "include/particle.h"
#include "include/constants.h"
#include <fstream>
#include <sys/stat.h>
#include "include/utils.h"
#include <getopt.h>
#include "include/Analyzer.h"
#include "include/analyzer_restart.h"

using namespace std;

 Analyzer::Analyzer(const csp_application::ObservationVector& time, const csp_application::ObservationVector& data, const std::string& constants_file, const std::string& output_folder,
             unsigned int column, const std::string& tag, unsigned int niter, const std::string& trainedPriorFile,
             bool append, unsigned int trim, bool verbose, const csp_application::ObservationVector& gtSpikes,
             bool has_trained_priors, bool has_gtspikes, unsigned int maxlen, const std::string& Gparam_file, int seed)
        : time(time), data(data), constants_file(constants_file), output_folder(output_folder), column(column), tag(tag),
          niter(niter), trainedPriorFile(trainedPriorFile), append(append), trim(trim), verbose(verbose),
          gtSpikes(gtSpikes), has_trained_priors(has_trained_priors), has_gtspikes(has_gtspikes),
          maxlen(maxlen), Gparam_file(Gparam_file),seed(seed){}

//For keeping a running list of time-independent pgas MC parameter estimates
void Analyzer::add_parameter_sample(std::vector<double> parameter_sample) {
    cout << "[DEBUG] Adding parameter_sample: ";
    for (const auto& val : parameter_sample) {
        cout << val << " ";
    }
    cout << endl;

    //parameter_samples.push_back(parameter_sample);
}


void Analyzer::run() {
    CSP_OBSERVE(csp_diagnostics::Scope numerical_run("Analyzer.run");)
    // Other init type stuff that was needed
    run_status = "running";
    run_message.clear();
    auto structural_failure = [&](const std::string& message) {
        run_status = "invalid_restart"; run_message = message;
        cerr << "[Analyzer.run] " << message << "; no output changes made" << endl;
    };

    // Original main function code here, replace argc/argv handling with member variables
    constpar constants(constants_file);
    constants.output_folder = output_folder;

    cout << "constants: " << constants_file << endl;
    cout << "output_folder: " << output_folder << endl;
    cout << "using column " << column << endl;
    cout << "using tsmode " << constants.TSMode << endl;

    ofstream parsamples;
    ofstream trajsamples;
    ofstream logp;
    if (niter > 0) constants.niter = niter;
    if (trim == 0 || constants.niter < 0) {
        structural_failure("trim must be positive and the target record count nonnegative"); return;
    }
    const std::uint64_t target_records = static_cast<std::uint64_t>(constants.niter);
    analyzer_restart::Record restart;
    if (append && !analyzer_restart::read(output_folder,tag,time.size(),restart,run_message)) {
        structural_failure(run_message); return;
    }
    const std::uint64_t existing_records = append ? restart.rows : 0;
    if (existing_records >= target_records) {
        run_status = "no_op"; run_message = "target saved-record count already met"; return;
    }
    // Validate the complete current-schema input before creating/opening any output.
    // Concurrent writers to this output/tag are not supported.
    struct stat sb;
    if (stat(output_folder.c_str(), &sb) != 0) {
        const int dir_err = mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            run_status="io_error"; run_message="cannot create output directory";
            cerr << "[Analyzer.run] " << run_message << endl; return;
        }
    }
    if (append) {
        parsamples.open(output_folder + "/param_samples_" + tag + ".dat", std::ios_base::app);
        trajsamples.open(output_folder + "/traj_samples_" + tag + ".dat", std::ios_base::app);
        logp.open(output_folder + "/logp_" + tag + ".dat", std::ios_base::app);
    } else {
        logp.open(output_folder + "/logp_" + tag + ".dat");
    }
    if (!logp.is_open() || (append && (!parsamples.is_open() || !trajsamples.is_open()))) {
        run_status="io_error"; run_message="cannot open output streams";
        cerr << "[Analyzer.run] " << run_message << endl; return;
    }

    param testpar(output_folder + "/param_samples_" + tag + ".dat");
    param testpar2(output_folder + "/param_samples_" + tag + ".dat");
		
    // loading the ground truth spikes
    // arma::vec gtSpikes; 
    if (has_gtspikes) {
        constants.KNOWN_SPIKES = true;
        //gtSpikes.load(gtSpike_file, arma::raw_ascii);
    }
    if (gtSpikes.size()==1){
        constants.KNOWN_SPIKES = false;
        has_gtspikes = false;
    }
    

    if (has_trained_priors) {
        string dum;
        // update the constants
        ifstream trainedPrior(trainedPriorFile);
        trainedPrior >> dum >> constants.G_tot_mean >> constants.G_tot_sd;
        trainedPrior >> dum >> constants.gamma_mean >> constants.gamma_sd;
        trainedPrior >> dum >> constants.DCaT_mean >> constants.DCaT_sd;
        trainedPrior >> dum >> constants.Rf_mean >> constants.Rf_sd;
        trainedPrior >> dum >> constants.alpha_sigma2 >> constants.beta_sigma2;
        trainedPrior >> dum >> constants.alpha_rate_b0 >> constants.beta_rate_b0;
        trainedPrior >> dum >> constants.alpha_rate_b1 >> constants.beta_rate_b1;
        trainedPrior >> dum >> constants.alpha_w01 >> constants.beta_w01;
        trainedPrior >> dum >> constants.alpha_w10 >> constants.beta_w10;
        trainedPrior.close();
    }

    if (niter > 0) {
        constants.niter = niter;
    }

    // Initialize the sampler (this will also reset the scales, that's why we need to initialize after we update the constants)
    // note the different constructors for SMC class here - one expects Analyzer to be called with a filename, the other takes data passed in directly

    SMC sampler(time, data, column, constants, false, seed, maxlen, Gparam_file);

    // Initialize the trajectory

    Trajectory traj_sam1(sampler.TIME, ""), traj_sam2(sampler.TIME, output_folder + "/traj_samples_" + tag + ".dat");
    
    for (unsigned int t = 0; t < sampler.TIME; ++t) {
        csp_application::trajectory_at(traj_sam1.B,t) = 0;
        csp_application::trajectory_at(traj_sam1.burst,t) = 0;
        csp_application::trajectory_at(traj_sam1.C,t) = 0;
        csp_application::trajectory_at(traj_sam1.S,t) = 0;
        if (has_gtspikes) csp_application::trajectory_at(traj_sam1.S,t) = csp_application::observation_at(gtSpikes,t);
        csp_application::trajectory_at(traj_sam1.Y,t) = 0;
    }
    // Set initial calcium value to first observation if specified in config
    if (constants.c0_is_first_y) {
        csp_application::trajectory_at(traj_sam1.C,0) = csp_application::observation_at(data,0);
    }

    // set initial parameters 
    if (append) {
        cout << "Parameter warm restart (RNG and trajectory are reinitialized)" << endl;
        testpar.G_tot = restart.values[0];
        testpar.gamma = restart.values[1];
        testpar.DCaT = restart.values[2];
        testpar.Rf = restart.values[3];
        testpar.gam_in = restart.values[4];
        testpar.gam_out = restart.values[5];
        testpar.sigma2 = restart.values[6];
        testpar.r0 = restart.values[7];
        testpar.r1 = restart.values[8];
        testpar.wbb[0] = restart.values[9];
        testpar.wbb[1] = restart.values[10];
    } else {
        cout << "Draw new parameters" << endl;
        testpar.r0 = constants.alpha_rate_b0 / constants.beta_rate_b0;
        testpar.r1 = constants.alpha_rate_b1 / constants.beta_rate_b1;
        testpar.wbb[0] = constants.alpha_w01 / constants.beta_w01;
        testpar.wbb[1] = constants.alpha_w10 / constants.beta_w10;
        testpar.sigma2 = constants.beta_sigma2 / (constants.alpha_sigma2 + 1); // the mode of the inverse gamma
        testpar.G_tot = constants.G_tot_mean;
        testpar.gamma = constants.gamma_mean;
        testpar.DCaT = constants.DCaT_mean;
        testpar.Rf = constants.Rf_mean;
        testpar.gam_in = constants.gam_in_mean;
        testpar.gam_out = constants.gam_out_mean;
    }
	
    cout << "start MCMC loop" << endl;
		

    // niter is the target TOTAL saved-record count. Keep zero-based saved indices;
    // perform trim existing sampler updates for each newly saved record.
    for (std::uint64_t i = existing_records * trim; i < target_records * trim; ++i) {
        sampler.PGAS(testpar, traj_sam1, traj_sam2);

        traj_sam1 = traj_sam2;
        for (unsigned int k = 0; k < 10; k++) {
            if (constants.SAMPLE_PARAMETERS) {
                sampler.sampleParameters(testpar, testpar2, traj_sam2);
            } else {
                testpar2 = testpar;
            }

            testpar = testpar2;
        }
				

        if ((i + 1) % trim == 0) {
            std::array<double,6> parameter_sample = {testpar.G_tot,
            testpar.gamma,
            testpar.DCaT,
            testpar.Rf,
            testpar.gam_in,
            testpar.gam_out};
            csp_application::append_parameter_row(parameter_samples, parameter_sample);
            testpar.write(parsamples, constants.sampling_frequency);
            traj_sam2.write(trajsamples, static_cast<unsigned int>((i + 1) / trim - 1));
            logp << testpar.logPrior(constants) + traj_sam2.logp(&testpar, &constants) << endl;
        }
        if (verbose) cout << "iteration:" << setw(5) << i << ", spikes: " << setw(5) << csp_application::sample_sum_compat(csp_application::TrajectorySamples(traj_sam1.S)) << '\r' << flush;
    }

    parsamples.close();
    trajsamples.close();

    // Save last param set for direct comparison b/t compiled c++ and python binding version
    ofstream lastParams(output_folder + "/last_params_" + tag + ".dat");
    lastParams << testpar.r0 << " " << testpar.r1 << " " << testpar.wbb[0] << " " << testpar.wbb[1] << " " << testpar.sigma2 << " "
               << testpar.G_tot << " " << testpar.gamma << ' ' << testpar.DCaT << ' ' << testpar.Rf << endl;
							 
		// Populate final_params after calculations
    final_params.push_back(testpar.G_tot);
    final_params.push_back(testpar.gamma);
    final_params.push_back(testpar.DCaT);
    final_params.push_back(testpar.Rf);
    final_params.push_back(testpar.gam_in);
    final_params.push_back(testpar.gam_out);
    run_status = "completed"; // Execution completion does not certify numerical validity.
}
