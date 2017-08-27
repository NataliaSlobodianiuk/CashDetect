#pragma once
// standart C++ libraries
#include<string>
#include<iostream>
#include<sstream>

/*
*  Function to show header
*/
void showConsoleHeader();


/* Function for getting int from console
*  Gets: message to show user before input
*  message to show user in case sth went wront
*  size - number for limit user input
*  Return: int - number from console
*/
int getIntCin(const char* message, const char* errorMessage, int min, int size);


/* Function for getting user`s option what to do next
*  Returns: int - option what to do
*/
int getUserOption();


/* Function for getting filename of image to be processed
*  Works until user type filename which is longer than 4 (dot and extension)
*  Return: filename
*/
std::string getFilename();