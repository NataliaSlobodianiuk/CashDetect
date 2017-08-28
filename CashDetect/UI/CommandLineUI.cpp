#include"CommandLineUI.h"


void showConsoleHeader() 
{
	std::cout << "################################################################################\n";
	std::cout << "  >>>                          Currency recognition                        <<<  \n";
	std::cout << "################################################################################\n\n";
}


int getIntCin(const char* message, const char* errorMessage, int min, int size) 
{
	int number;
	do {
		std::cout << message << std::flush;

		std::string numberStr;
		getline(std::cin, numberStr);

		std::stringstream strstream(numberStr);

		if (strstream >> number) {
			if (number >= min && number < size)
				break;
			else
				std::cout << errorMessage << std::endl;
		}
		else {
			std::cout << errorMessage << std::endl;
		}

	} while (true);

	return number;
}


int getUserOption() 
{
	std::cout << " ## Menu:\n";
	std::cout << "   1 - Recognize currency using SIFT feature transform\n";
	std::cout << "   2 - Recognize coins\n";
	std::cout << "   0 - Exit\n";

	return getIntCin("\n >> Option [0, 2]: ", "Select one of the options above!", 0, 4);
}


std::string getFilename()
{
	// Filename for image to be read
	std::string filename;

	// Get filename from user
	filename = "";
	do {
		std::cout << "  >> Path to file inside data\\examples folder: ";
		// Get filename from user
		getline(std::cin, filename);

		if (filename == "")
		{
			std::cerr << "  => File path can't be empty!\n" << std::endl;
		}
	} while (filename.size() < 5);

	return filename;
}