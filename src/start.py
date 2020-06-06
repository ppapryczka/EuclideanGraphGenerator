from datetime import datetime

from graph_generation import generate_point_positions, generate_simple_random_graph, generate_graphs
from graph_metrics import calculate_real_mean_degree, calculate_expected_mean_degree, calculate_approximation_error, calculate_real_number_of_edges, calculate_expected_number_of_edges, calculate_real_density, calculate_expected_density, \
    calculate_expected_degree_distribution, calculate_real_degree_distribution, calculate_graph_components_statistics, calculate_graph_cycle_related_statistics, calculate_mean_clustering, calculate_degree_distribution_difference
from graph_visualization import show_graph, create_output_directory, show_distribution_chart


def main():
    # Number of graphs to generate with given parameters
    number_of_graphs = 100
    # Number of vertices
    n = 200
    # Euclidean network radius
    r = 0.1

    output_directory = create_output_directory("generation_{}".format(datetime.now()))

    perform_long_operations = True

    # -------------------------------------------------------------------------------------------------
    # Test graph generation
    print("Liczba wierzchołków: {}".format(n))
    print("Zasięg:              {}".format(r))
    print()

    # Generate point positions
    print(">>>Generowanie pozycji wierzchołków")
    positions, vertex_execution_time = generate_point_positions(0.0, 1.0, n)
    print("Ukończone w          {} s".format(vertex_execution_time))
    print()

    # Generate Euclidean graph using kd tree
    print(">>>Generowanie grafu Euklidesowego (drzewo kd)")
    g_kd, kd_execution_time = generate_simple_random_graph(
        number_of_vertices=n,
        radius=r,
        positions=positions,
        use_kd_tree=True
    )
    print("Liczba krawędzi:     {}".format(len(g_kd.edges)))
    print("Ukończone w          {} s".format(kd_execution_time))
    print()

    # Generate Euclidean graph using n^2 check
    print(">>>Generowanie grafu Euklidesowego (n^2)")
    g_n2, n2_execution_time = generate_simple_random_graph(
        number_of_vertices=n,
        radius=r,
        positions=positions,
        use_kd_tree=False
    )
    print("Liczba krawędzi:     {}".format(len(g_n2.edges)))
    print("Ukończone w          {} s".format(n2_execution_time))
    print()

    # Show graph and save it to file
    show_graph(g_kd, output_directory + "/sample_graph_{}_{}.png".format(n, r))

    # -------------------------------------------------------------------------------------------------
    # Checking statistical properties of graphs
    print(">>>Sprawdzanie właściwości statystycznych dla {} grafów".format(number_of_graphs))
    print(">>>Generacja grafów ...")
    graphs, graphs_generation_time = generate_graphs(
        number_of_graphs=number_of_graphs,
        number_of_vertices=n,
        radius=r,
        use_kd_tree=True
    )
    print("Ukończone w                   {} s".format(graphs_generation_time))
    print()

    print(">>>Sprawdzanie średniego stopnia wierzchołka")
    real_mean_degree = calculate_real_mean_degree(graphs)
    expected_mean_degree = calculate_expected_mean_degree(n, r)
    mean_degree_error = calculate_approximation_error(real_mean_degree, expected_mean_degree)
    print("Rzeczywisty średni stopień    {}".format(real_mean_degree))
    print("Przewidywany średni stopień   {}".format(expected_mean_degree))
    print("Błąd                          {} %".format(mean_degree_error))
    print()

    print(">>>Sprawdzanie ilości krawędzi")
    real_number_of_edges = calculate_real_number_of_edges(graphs)
    expected_number_of_edges = calculate_expected_number_of_edges(n, r)
    number_of_edges_error = calculate_approximation_error(real_number_of_edges, expected_number_of_edges)
    print("Rzeczywista ilość krawędzi    {}".format(real_number_of_edges))
    print("Przewidywana ilość krawędzi   {}".format(expected_number_of_edges))
    print("Błąd                          {} %".format(number_of_edges_error))
    print()

    print(">>>Sprawdzanie gęstości grafu ...")
    real_density = calculate_real_density(graphs)
    expected_density = calculate_expected_density(n, r)
    density_error = calculate_approximation_error(real_density, expected_density)
    print("Rzeczywista gęstość           {}".format(real_density))
    print("Przewidywana gęstość          {}".format(expected_density))
    print("Błąd                          {} %".format(density_error))
    print()

    print(">>>Sprawdzanie rozkładu stopni wierzchołków ...")
    expected_distribution = calculate_expected_degree_distribution(n, r)
    real_distribution = calculate_real_degree_distribution(n, r, graphs)
    difference = calculate_degree_distribution_difference(real_distribution, expected_distribution)
    max_value = max(list(map(lambda v: v[1], expected_distribution + real_distribution + difference)))
    min_value = min(list(map(lambda v: v[1], expected_distribution + real_distribution + difference)))
    y_range = min_value, max_value
    print(">>>Generowanie wykresu rozkładu oczekiwanego ...")
    show_distribution_chart(expected_distribution, y_range, output_directory + "/chart_expected_{}_{}.png".format(n, r))
    print(">>>Generowanie wykresu rozkładu rzeczywistego ...")
    show_distribution_chart(real_distribution, y_range, output_directory + "/chart_real_{}_{}_{}.png".format(n, r, number_of_graphs))
    print(">>>Generowanie różnicy rozkładów ...")
    show_distribution_chart(difference, y_range, output_directory + "/chart_difference_{}_{}_{}.png".format(n, r, number_of_graphs))
    print()

    if perform_long_operations:
        # print("-------------------------------------------------------------------")
        print("Sprawdzanie statystyk dotyczących składowych ...")
        component_stats = calculate_graph_components_statistics(graphs)
        print("Średnia liczba składowych spójnych:     {}".format(component_stats.mean_number_of_components))
        print("Średnia liczba wierzchołków składowej:  {}".format(component_stats.mean_number_of_nodes_in_component))
        print("Średnia liczba krawędzi składowej:      {}".format(component_stats.mean_number_of_edges_in_component))
        print("Średnia gęstość składowej:              {}".format(component_stats.mean_density_of_component))
        print("Średnia liczba drzew w grafie:          {}".format(component_stats.mean_number_of_trees_per_graph))
        print("Średnia liczba wierzchołków w drzewie:  {}".format(component_stats.mean_size_of_tree))
        print()

        # print("-------------------------------------------------------------------")
        print("Sprawdzanie statystyk dotyczących cykli ...")
        cycle_stats = calculate_graph_cycle_related_statistics(graphs)
        print("Liczba cykli bazowych:             {}".format(cycle_stats.mean_number_of_base_cycles))
        print("Średnia długość cyklu:             {}".format(cycle_stats.mean_length_of_base_cycle))
        print()

        print("Sprawdzanie klasteryzacji ...")
        mean_clustering = calculate_mean_clustering(graphs)
        print()

    output = open(output_directory + '/result_{}_{}_{}.txt'.format(n, r, number_of_graphs), 'w+')
    print(number_of_graphs, file=output)
    print(n, file=output)
    print(r, file=output)
    print(kd_execution_time, file=output)
    print(len(g_kd.edges), file=output)
    print(n2_execution_time, file=output)
    print(len(g_n2.edges), file=output)
    print(real_mean_degree, file=output)
    print(expected_mean_degree, file=output)
    print(real_number_of_edges, file=output)
    print(expected_number_of_edges, file=output)
    print(real_density, file=output)
    print(expected_density, file=output)
    print(density_error, file=output)
    if perform_long_operations:
        print(component_stats.mean_number_of_components, file=output)
        print(component_stats.mean_number_of_nodes_in_component, file=output)
        print(component_stats.mean_number_of_edges_in_component, file=output)
        print(component_stats.mean_density_of_component, file=output)
        print(component_stats.mean_number_of_trees_per_graph, file=output)
        print(component_stats.mean_size_of_tree, file=output)
        print(cycle_stats.mean_number_of_base_cycles, file=output)
        print(cycle_stats.mean_length_of_base_cycle, file=output)
        print(mean_clustering, file=output)
    output.close()


if __name__ == "__main__":
    main()
