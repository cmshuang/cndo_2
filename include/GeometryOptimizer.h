#ifndef GEOMETRY_OPTIMIZER_H
#define GEOMETRY_OPTIMIZER_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <armadillo>
#include <string>

#include "Molecule.h"

class GeometryOptimizer {
 public:
    void optimize();
    double linesearch_wolfe(arma::vec& direction);
    //Constructor
    GeometryOptimizer(Molecule* mol, const string& method) {
        if (method != "bfgs" && method != "sd") {
            throw std::invalid_argument("Geometry optimizer can currently only run BFGS and steepest descent.");
        }
        m_method = method;
        m_mol = mol;
        std::vector<Atom>& atoms = m_mol->get_atoms();

        int num_atoms = atoms.size();

        m_pos = arma::mat(3, num_atoms);

        for (int i = 0; i < atoms.size(); i++) {
            m_pos.col(i) = atoms[i].get_R();
        }

        //m_pos.print("positions");
    }

    arma::mat get_pos() const {
        /* Getter function for atom positions
         */
        return m_pos;
    }

 private:
    void run_steepest_descent();
    void run_bfgs();
    double m_tol=1E-3;
    double m_stepsize=0.1;
    int m_max_iter = 1E+3;
    string m_method;
    Molecule* m_mol;
    arma::mat m_pos;
};

void put_coords_in_atoms(const arma::mat& pos, Molecule* mol);

#endif //GEOMETRY_OPTIMIZER_H