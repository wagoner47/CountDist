#include <argp.h>


struct arguments {
	char * args[1];  // PARAMETER_FILE
	bool test;       // -t, --test flag
};

static struct argp_option options[] = {
	{"test", 't', 0, 0, "Run the code in test mode"},
	{0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
	auto *arguments = (struct arguments*) state->input;

	switch(key) {
		case 't':
			arguments->test = true;
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

static char args_doc[] = "PARAMETER_FILE";

static char doc[] = "Calculate separations between galaxies and save in files";

static struct argp argp = {options, parse_opt, args_doc, doc};
