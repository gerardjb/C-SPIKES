#include <iostream>
#include <iomanip>
#include "include/particle.h"
#include "include/constants.h"
#include <fstream>
#include <sys/stat.h>
#include "include/utils.h"
#include <getopt.h>
#include "include/Analyzer.h"

using namespace std;

 Analyzer::Analyzer(const arma::vec& time, const arma::vec& data, const std::string& constants_file, const std::string& output_folder,
             unsigned int column, const std::string& tag, unsigned int niter, const std::string& trainedPriorFile,
             bool init_old, unsigned int trim, bool verbose, const arma::vec& gtSpikes,
             bool has_trained_priors, bool has_gtspikes, unsigned int maxlen, const std::string& Gparam_file, int seed, const std::string& old_tag)
        : time(time), data(data), constants_file(constants_file), output_folder(output_folder), column(column), tag(tag),
          niter(niter), trainedPriorFile(trainedPriorFile), init_old(init_old), trim(trim), verbose(verbose),
          gtSpikes(gtSpikes), has_trained_priors(has_trained_priors), has_gtspikes(has_gtspikes),
          maxlen(maxlen), Gparam_file(Gparam_file),seed(seed), old_tag(old_tag){}

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
    // Other init type stuff that was needed
    int existing_samples=0;
    int existing_trajs=0;

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
    istringstream last_params;
    istringstream last_state;

    struct stat sb;
    if (stat(output_folder.c_str(), &sb) != 0) {
        const int dir_err = mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            printf("Error creating directory!");
            exit(1);
        }
    }

    if (init_old) {
        // Handling the last entry for the time-invariant paramters
        ifstream ps(output_folder + "/param_samples_" + old_tag + ".dat");
        string line;        
        while (ps >> std::ws && std::getline(ps, line)) existing_samples++;

        if (existing_samples == 0) {
            cerr << "empty existing file!!" << endl;
            cerr << "Filename was " + output_folder + "/param_samples_" + old_tag + ".dat" << endl;
            exit(1);
        }
        
        last_params.str(line); 
        ps.close();

        // Handling the last entry for the system state
        ifstream traj_file(output_folder + "/traj_samples_" + old_tag + ".dat");// <- stub, add to constructor and bindings
        string last_line;
        while (traj_file >> std::ws && std::getline(traj_file, last_line)) existing_trajs++;
        if (existing_trajs==0) {
            cerr << "Error: no valid data lines in " 
                << output_folder << "/traj_samples_" << old_tag << ".dat\n";
            exit(EXIT_FAILURE);
        }
        last_state.str(last_line);
        traj_file.close();

        logp.open(output_folder + "/logp_" + tag + ".dat", std::ios_base::app);
    } else {
        logp.open(output_folder + "/logp_" + tag + ".dat");
    }

    param testpar(output_folder + "/param_samples_" + tag + ".dat");
    param testpar2(output_folder + "/param_samples_" + tag + ".dat");
		
    // loading the ground truth spikes <- truned this off - use python interface
    // arma::vec gtSpikes; 
    if (has_gtspikes) {
        constants.KNOWN_SPIKES = true;
        //gtSpikes.load(gtSpike_file, arma::raw_ascii);
    }
    if (gtSpikes.n_elem==1){
        constants.KNOWN_SPIKES = false;
        has_gtspikes = false;
    }

    if (niter > 0) {
        constants.niter = niter;
    }

    // Initialize the sampler (this will also reset the scales, that's why we need to initialize after we update the constants)
    // note the different constructors for SMC class here - one expects Analyzer to be called with a filename, the other takes data passed in directly

    SMC sampler(time, data, column, constants, false, seed, maxlen, Gparam_file);
    cout<<"data size = "<<data.n_elem<<endl;
    cout<<"first two data points = "<<data(0)<<" "<<data(1)<<endl;

    // Initialize the trajectory

    Trajectory traj_sam1(sampler.TIME, ""), traj_sam2(sampler.TIME, output_folder + "/traj_samples_" + tag + ".dat");

    // set initial parameters 
    if (init_old) {
        cout << "Use parameters and states from old_tag "<<old_tag<<" run" << endl;
        // Loading the last parameters from previous session
        vector<string> parse_params;
        while (last_params.good()) {
            string substr;
            getline(last_params, substr, ',');
            parse_params.push_back(substr);
        }

        testpar.G_tot = stod(parse_params[0]);
        testpar.gamma = stod(parse_params[1]);
        testpar.DCaT = stod(parse_params[2]);
        testpar.Rf = stod(parse_params[3]);
        testpar.gam_in = stod(parse_params[4]);
        testpar.gam_out = stod(parse_params[5]);
        testpar.ca_half = stod(parse_params[6]);
        testpar.n_gate = stod(parse_params[7]);
        testpar.sigma2 = stod(parse_params[8]);
        testpar.r0 = stod(parse_params[9]);
        testpar.r1 = stod(parse_params[10]);
        testpar.wbb[0] = stod(parse_params[11]);
        testpar.wbb[1] = stod(parse_params[12]);

        cout << "Loaded parameters: G_tot=" << testpar.G_tot << ", gamma=" << testpar.gamma
             << ", DCaT=" << testpar.DCaT << ", Rf=" << testpar.Rf
             << ", gam_in=" << testpar.gam_in << ", gam_out=" << testpar.gam_out
             << ", ca_half=" << testpar.ca_half << ", n_gate=" << testpar.n_gate
             << ", sigma2=" << testpar.sigma2
             << ", r0=" << testpar.r0 << ", r1=" << testpar.r1
             << ", wbb[0]=" << testpar.wbb[0] << ", wbb[1]=" << testpar.wbb[1] << endl;

        // Initialize trajectory with old session state values
        vector<string> tokens;
        while (last_state.good()) {
            string token;
            getline(last_state, token, ',');
            if (token.empty()) continue; // Skip empty tokens
            tokens.push_back(token);
        }

        // And initiaizing the trajectory
        traj_sam1.B(0) = stod(tokens[2]);
        traj_sam1.burst(0) = stod(tokens[1]);
        traj_sam1.C(0) = stod(tokens[4]);
        traj_sam1.S(0) = stod(tokens[3]);
        traj_sam1.Y(0) = 0;

        cout<< "Initial values: B=" << traj_sam1.B(0) << ", burst=" << traj_sam1.burst(0)
            << ", C=" << traj_sam1.C(0) << ", S=" << traj_sam1.S(0) << endl;

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
        testpar.ca_half = constants.ca_half_mean;
        testpar.n_gate = constants.n_gate_mean;

        cout << "Loaded parameters: G_tot=" << testpar.G_tot << ", gamma=" << testpar.gamma
             << ", DCaT=" << testpar.DCaT << ", Rf=" << testpar.Rf
             << ", gam_in=" << testpar.gam_in << ", gam_out=" << testpar.gam_out
             << ", c_half=" << testpar.ca_half << ", n_gate=" << testpar.n_gate
             << ", sigma2=" << testpar.sigma2
             << ", r0=" << testpar.r0 << ", r1=" << testpar.r1
             << ", wbb[0]=" << testpar.wbb[0] << ", wbb[1]=" << testpar.wbb[1] << endl;

        // Initiaizing the trajectory
        traj_sam1.B(0) = 0;
        traj_sam1.burst(0) = 0;
        traj_sam1.C(0) = 0;
        traj_sam1.S(0) = 0;
        traj_sam1.Y(0) = 0;

        cout<< "Initial values: B=" << traj_sam1.B(0) << ", burst=" << traj_sam1.burst(0)
            << ", C=" << traj_sam1.C(0) << ", S=" << traj_sam1.S(0) << endl;
    }

    // Initialiaze the remainder of the trajectory <- need to check this affects the "early false spike effect", else, might need longer intialization
    // Looks like this worked, might consider extending the initialization chunk with some rubric
    for (unsigned int t = 1; t < sampler.TIME; ++t) {
        traj_sam1.B(t) = 0;
        traj_sam1.burst(t) = 0;
        traj_sam1.C(t) = 0;
        traj_sam1.S(t) = 0;
        if (has_gtspikes) traj_sam1.S(t) = gtSpikes(t);
        traj_sam1.Y(t) = 0;
    }
	
    cout << "start MCMC loop" << endl;

    for (unsigned int i = 0; i < constants.niter; i++) {
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
				

        if (i % trim == 0) {
            std::vector<double> parameter_sample = {testpar.G_tot,
            testpar.gamma,
            testpar.DCaT,
            testpar.Rf,
            testpar.gam_in,
            testpar.gam_out,
            testpar.ca_half,
            testpar.n_gate};
            arma::rowvec new_row(parameter_sample);
            if (parameter_samples.n_cols==0){
                parameter_samples = arma::mat(new_row);
            }
            else{
                parameter_samples = arma::join_cols(parameter_samples, arma::mat(new_row));
            }
            testpar.write(parsamples, constants.sampling_frequency);
            traj_sam2.write(trajsamples, i / trim);
            logp << testpar.logPrior(constants) + traj_sam2.logp(&testpar, &constants) << endl;
        }
        if (verbose) cout << "iteration:" << setw(5) << i << ", spikes: " << setw(5) << arma::accu(traj_sam1.S) << '\r' << flush;
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
    final_params.push_back(testpar.ca_half);
    final_params.push_back(testpar.n_gate);
}
