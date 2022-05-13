#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "SCFFunctions.h"
#include "GeometryOptimizer.h"

using namespace std;

int main(int argc, char* argv[])
{

  if (argc == 1) {
    printf("usage ./final filename --diis[optional] --bfgs[optional], for example ./final example.txt --diis --bfgs");
    return EXIT_FAILURE;
  }

  std::vector <std::string> all_args;
  all_args.assign(argv + 1, argv + argc);

  string fname(all_args[0]);

  bool toDIIS = false;
  std::string method = "sd";

  for (auto iter=all_args.begin(); iter != all_args.end(); iter++) {
    auto & arg = * iter;
    if (arg == "--diis")
      toDIIS = true;
    if (arg == "--bfgs")
      method = "bfgs";
  }

  cout << "Reading molecule from " << fname << endl;

  // Parse input file, create molecule
  try {
    Molecule my_molecule = read_molecule(fname, toDIIS);
    // // To print information about each basis function
    // vector<Atom> atoms = my_molecule.get_atoms();
    // for (int i = 0; i < atoms.size(); i++) {
    //   vector<BasisFunction>& basis_functions = atoms[i].get_basis_functions();
    //   for (int j = 0; j < basis_functions.size(); j++) {
    //     basis_functions[j].print_info();
    //   }
    // }

    // for (int i = 0; i < my_molecule.get_all_basis_functions().size(); i++) {
    //    my_molecule.get_all_basis_functions()[i]->print_info();
    // }

    GeometryOptimizer my_optimizer(&my_molecule, method);

    my_optimizer.optimize();

    my_molecule.write_output("output.txt");

    // // To print information about each basis function
    // vector<Atom> atoms = my_molecule.get_atoms();
    // for (int i = 0; i < atoms.size(); i++) {
    //   vector<BasisFunction>& basis_functions = atoms[i].get_basis_functions();
    //   for (int j = 0; j < basis_functions.size(); j++) {
    //     basis_functions[j].print_info();
    //   }
    // }
  }
  catch (std::invalid_argument &err) {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}