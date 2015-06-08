#pragma once

#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cstring>
#include <csignal>
#include <functional>

/* adapted from http://stackoverflow.com/a/25087408 */
class StdioHandler
{
private:
	pid_t pid = 0;
	int origfd;
	int streamid;
	int pipefd[2];
public:
	enum class Stream
	{
		stdout = STDOUT_FILENO,
		stderr = STDERR_FILENO
	};
	StdioHandler(Stream stream, std::function<void(const char*)> callback);
	~StdioHandler();
};

class StdOutHandler : public StdioHandler
{
public:
	StdOutHandler(std::function<void(const char*)> callback);
};

class StdErrHandler : public StdioHandler
{
public:
	StdErrHandler(std::function<void(const char*)> callback);
};
