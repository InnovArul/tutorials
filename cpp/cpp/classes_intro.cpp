#include<iostream>
using namespace std;

struct student
{
	char* name;
	int rollnumber;
	int marks[10];
};

class studentclass
{
	char* name;
	int rollnumber;
	int* marks;

public:
	studentclass(char* name, int rollnumber, int* marks)
	{
		this->name = name;
		this->rollnumber = rollnumber;
		this->marks = marks;
	}

	int get_rollnumber()
	{
		return this->rollnumber;
	}

	~studentclass()
	{
		cout << "destructor called";
	}
};



int main()
{
	int marks[] = { 1,2,3,4,5,6,7 };
	int rollnumber = 100;
	char *name = "student";

	studentclass* student1 = new studentclass(name, rollnumber, marks);
	cout << student1->get_rollnumber();

	return 0;
}