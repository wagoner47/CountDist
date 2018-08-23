#include <argp.h>
#include "logging.h"

const char *argp_program_version = "countdist2 0.1";
static char doc[] = "Calculate separations between objects and save in a database\vThe parameter file argument is the only non-optional parameter, and this should be the path to a configuration file containing the needed parameters, such as catalog file names, output database file path, and separation limits. See full documentation (todo) for more details on the configuration file. Also, the verbosity level should be set by integer representation for the following available levels: debug(0), info(1), warning(2), error(3), fatal(4).";
static char args_doc[] = "PARAMETER_FILE";


struct arguments {
	char * args[1];        // PARAMETER_FILE
	bool test;             // -t, --test (flag)
	severity_level level;  // -l, --level LEVEL
};

static struct argp_option options[] = {
	{"test", 't', 0, 0, "Run the code in test mode (currently does nothing)"}, 
	{"level", 'l', "LEVEL", 0, "Set verbosity level (default is 10=fatal)"},
	{0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
	auto *arguments = (struct arguments*) state->input;

	switch(key) {
		case 't':
			arguments->test = true;
			break;
	        case 'l':
		        arguments->level = static_cast<severity_level>(atoi(arg));
			break;
		case ARGP_KEY_ARG:
			if (state->arg_num >= 1) {
				argp_usage(state);
			}
			arguments->args[state->arg_num] = arg;
			break;
		case ARGP_KEY_END:
			if (state->arg_num < 1) {
				argp_usage(state);
			}
			break;
		default:
			return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
