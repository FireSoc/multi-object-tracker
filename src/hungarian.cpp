#include "hungarian.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

std::vector<int> solve_wide(const HungarianAlgorithm::CostMatrix& costs) {
    const std::size_t row_count = costs.size();
    const std::size_t column_count = costs.front().size();

    // Index zero is the augmenting-path sentinel. Real rows and columns are one-based.
    std::vector<long double> row_potential(row_count + 1, 0.0L);
    std::vector<long double> column_potential(column_count + 1, 0.0L);
    std::vector<std::size_t> matched_row(column_count + 1, 0);
    std::vector<std::size_t> predecessor(column_count + 1, 0);

    for (std::size_t row = 1; row <= row_count; ++row) {
        matched_row[0] = row;
        std::size_t current_column = 0;
        std::vector<long double> minimum_slack(column_count + 1,
                                               std::numeric_limits<long double>::infinity());
        std::vector<bool> visited(column_count + 1, false);

        do {
            visited[current_column] = true;
            const std::size_t current_row = matched_row[current_column];
            long double smallest_slack = std::numeric_limits<long double>::infinity();
            std::size_t next_column = 0;

            for (std::size_t column = 1; column <= column_count; ++column) {
                if (visited[column]) {
                    continue;
                }

                const long double reduced_cost =
                    static_cast<long double>(costs[current_row - 1][column - 1]) -
                    row_potential[current_row] - column_potential[column];
                if (reduced_cost < minimum_slack[column]) {
                    minimum_slack[column] = reduced_cost;
                    predecessor[column] = current_column;
                }
                if (minimum_slack[column] < smallest_slack) {
                    smallest_slack = minimum_slack[column];
                    next_column = column;
                }
            }

            for (std::size_t column = 0; column <= column_count; ++column) {
                if (visited[column]) {
                    row_potential[matched_row[column]] += smallest_slack;
                    column_potential[column] -= smallest_slack;
                } else {
                    minimum_slack[column] -= smallest_slack;
                }
            }
            current_column = next_column;
        } while (matched_row[current_column] != 0);

        do {
            const std::size_t previous_column = predecessor[current_column];
            matched_row[current_column] = matched_row[previous_column];
            current_column = previous_column;
        } while (current_column != 0);
    }

    std::vector<int> assignment(row_count, -1);
    for (std::size_t column = 1; column <= column_count; ++column) {
        if (matched_row[column] != 0) {
            assignment[matched_row[column] - 1] = static_cast<int>(column - 1);
        }
    }
    return assignment;
}

}  // namespace

std::vector<int> HungarianAlgorithm::solve(const CostMatrix& costs) {
    if (costs.empty()) {
        return {};
    }

    const std::size_t row_count = costs.size();
    const std::size_t column_count = costs.front().size();
    for (const auto& row : costs) {
        if (row.size() != column_count) {
            throw std::invalid_argument("Hungarian cost matrix must be rectangular");
        }
        for (const double cost : row) {
            if (!std::isfinite(cost)) {
                throw std::invalid_argument("Hungarian costs must be finite");
            }
        }
    }

    if (column_count == 0) {
        return std::vector<int>(row_count, -1);
    }
    if (row_count <= column_count) {
        return solve_wide(costs);
    }

    CostMatrix transposed(column_count, std::vector<double>(row_count));
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t column = 0; column < column_count; ++column) {
            transposed[column][row] = costs[row][column];
        }
    }

    const std::vector<int> transposed_assignment = solve_wide(transposed);
    std::vector<int> assignment(row_count, -1);
    for (std::size_t original_column = 0; original_column < column_count; ++original_column) {
        const int original_row = transposed_assignment[original_column];
        assignment[static_cast<std::size_t>(original_row)] = static_cast<int>(original_column);
    }
    return assignment;
}
