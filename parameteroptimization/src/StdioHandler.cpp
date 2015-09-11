#include "StdioHandler.h"


StdioHandler::StdioHandler(StdioHandler::Stream stream, std::function<void (const char *)> callback)
	:streamid(static_cast<int>(stream))
{
	origfd = dup(streamid);

	pipe(pipefd); // create pipe
	pid = fork(); //spawn a child process to handle output of pipe
	if (pid == 0)
	{
		char line[256];
		FILE* output;

		close(pipefd[1]);
		output = fdopen(pipefd[0], "r");
		if (output)
		{
			while(fgets(line, sizeof(line), output))
			{

				int n = strlen(line);
				if (n > 0)
					if (line[n-1] == '\n') line[n-1] = 0;
				callback(line);
			}
			fclose(output);
		}
		abort();
	} else {
		// connect input of pipe to
		close(pipefd[0]);
		dup2(pipefd[1], streamid);
	}
}

StdioHandler::~StdioHandler()
{
	int status;

	usleep(10000);

	close(pipefd[1]);
	kill(pid,SIGINT);

	waitpid(pid, &status, 0);

	dup2(origfd, streamid);
}


StdOutHandler::StdOutHandler(std::function<void (const char *)> callback) :
	StdioHandler(Stream::stdout, callback)
{
}


StdErrHandler::StdErrHandler(std::function<void (const char *)> callback) :
	StdioHandler(Stream::stderr, callback)
{
}
