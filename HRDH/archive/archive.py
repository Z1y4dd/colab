def create_3d_correlation_explorer(df_all, top_correlations, save_path='imgs/'):
    """
    Create interactive 3D scatter plots showing relationships between 3 variables at once.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Get top 3 correlations
    top_3 = []
    for n_wells in [4, 3, 2]:
        if n_wells in correlations_by_well_count:
            for pair, wells_data, info in correlations_by_well_count[n_wells][:2]:
                top_3.append((pair, info['avg_abs_corr']))
    
    # Find variables that appear most frequently
    var_counts = {}
    for (log_var, lab_var), _ in top_3[:6]:
        var_counts[log_var] = var_counts.get(log_var, 0) + 1
        var_counts[lab_var] = var_counts.get(lab_var, 0) + 1
    
    # Get top 3 most connected variables
    top_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if len(top_vars) >= 3:
        var1, var2, var3 = [v[0] for v in top_vars]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        colors = {'HRDH_697': 'blue', 'HRDH_1119': 'orange', 
                  'HRDH_1804': 'green', 'HRDH_1867': 'red'}
        
        for well in df_all['Well'].unique():
            well_data = df_all[df_all['Well'] == well]
            valid_data = well_data[[var1, var2, var3]].dropna()
            
            fig.add_trace(go.Scatter3d(
                x=valid_data[var1],
                y=valid_data[var2],
                z=valid_data[var3],
                mode='markers',
                name=well,
                marker=dict(
                    size=4,
                    color=colors.get(well, 'gray'),
                    opacity=0.6
                ),
                text=well_data.index,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              f'{var1}: %{{x:.2f}}<br>' +
                              f'{var2}: %{{y:.2f}}<br>' +
                              f'{var3}: %{{z:.2f}}<br>' +
                              'Index: %{text}'
            ))
        
        fig.update_layout(
            title=f'3D Correlation Explorer: {var1.replace("Log_", "").replace("Lab_", "")} vs ' +
                  f'{var2.replace("Log_", "").replace("Lab_", "")} vs {var3.replace("Log_", "").replace("Lab_", "")}',
            scene=dict(
                xaxis_title=var1.replace('Log_', '').replace('Lab_', ''),
                yaxis_title=var2.replace('Log_', '').replace('Lab_', ''),
                zaxis_title=var3.replace('Log_', '').replace('Lab_', ''),
            ),
            height=800
        )
        
        fig.write_html(f'{save_path}3d_correlation_explorer.html')
        fig.show()

def create_correlation_network(df_all, correlations_by_well_count, min_correlation=0.6, save_path='imgs/'):
    """
    Create an interactive network graph showing relationships between variables.
    """
    import networkx as nx
    import plotly.graph_objects as go
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges
    for n_wells in [4, 3, 2]:
        if n_wells in correlations_by_well_count:
            for pair, wells_data, info in correlations_by_well_count[n_wells]:
                if info['avg_abs_corr'] >= min_correlation:
                    log_var, lab_var = pair
                    
                    # Add nodes
                    G.add_node(log_var, node_type='log')
                    G.add_node(lab_var, node_type='lab')
                    
                    # Add edge with weight based on correlation strength
                    G.add_edge(log_var, lab_var, 
                              weight=info['avg_abs_corr'],
                              n_wells=n_wells,
                              correlation_type=info['correlation_type'])
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create Plotly figure
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Color based on correlation type
        color = 'green' if edge[2]['correlation_type'] == 'Positive' else 'red'
        width = edge[2]['weight'] * 5
        
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"{edge[0].replace('Log_', '')} - {edge[1].replace('Lab_', '')}<br>" +
                 f"Avg |r|: {edge[2]['weight']:.3f}<br>" +
                 f"Wells: {edge[2]['n_wells']}",
            showlegend=False
        ))
    
    # Node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[node.replace('Log_', '').replace('Lab_', '') for node in G.nodes()],
        textposition="top center",
        marker=dict(
            size=15,
            color=['lightblue' if G.nodes[node]['node_type'] == 'log' else 'lightcoral' 
                   for node in G.nodes()],
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext=[f"{node}<br>Type: {G.nodes[node]['node_type']}<br>" +
                   f"Connections: {G.degree(node)}" for node in G.nodes()]
    )
    
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        title='Correlation Network Graph (|r| ≥ 0.6)',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800
    )
    
    fig.write_html(f'{save_path}correlation_network.html')
    fig.show()

def create_parallel_coordinates(df_all, selected_vars, save_path='imgs/'):
    """
    Create parallel coordinates plot to visualize multi-dimensional patterns.
    """
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
    
    # Prepare data
    plot_data = df_all[selected_vars + ['Well']].dropna()
    
    # Standardize numerical data
    scaler = StandardScaler()
    scaled_data = plot_data.copy()
    scaled_data[selected_vars] = scaler.fit_transform(plot_data[selected_vars])
    
    # Create color mapping for wells
    well_colors = {'HRDH_697': 0, 'HRDH_1119': 1, 'HRDH_1804': 2, 'HRDH_1867': 3}
    scaled_data['Well_Color'] = scaled_data['Well'].map(well_colors)
    
    # Create dimensions
    dimensions = []
    for var in selected_vars:
        dimensions.append(
            dict(
                label=var.replace('Log_', '').replace('Lab_', ''),
                values=scaled_data[var],
                range=[scaled_data[var].min(), scaled_data[var].max()]
            )
        )
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=scaled_data['Well_Color'],
                colorscale=[[0, 'blue'], [0.33, 'orange'], [0.66, 'green'], [1, 'red']],
                showscale=True,
                colorbar=dict(
                    title='Well',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['697', '1119', '1804', '1867']
                )
            ),
            dimensions=dimensions
        )
    )
    
    fig.update_layout(
        title='Parallel Coordinates: Multi-Variable Patterns Across Wells',
        height=600
    )
    
    fig.write_html(f'{save_path}parallel_coordinates.html')
    fig.show()

def create_hierarchical_clustering_heatmap(df_all, variables, save_path='imgs/'):
    """
    Create a clustered heatmap showing how samples group together.
    """
    import seaborn as sns
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    
    # Prepare data
    data_for_clustering = df_all[variables].dropna()
    well_labels = df_all.loc[data_for_clustering.index, 'Well']
    
    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), 
                                   gridspec_kw={'width_ratios': [1, 4]})
    
    # Create color map for wells
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#ff7f0e', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#d62728'
    }
    
    # Create well color array
    colors = [well_colors[well] for well in well_labels]
    
    # Plot well indicator
    well_matrix = [[1] for _ in colors]
    # cmap_color = sns.color_palette(list(well_colors.values()))
    ax1.imshow(well_matrix, aspect='auto', cmap='RdYlGn')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Well', fontsize=12)
    
    # Create clustered heatmap
    g = sns.clustermap(scaled_data, 
                       col_cluster=True,
                       row_cluster=True,
                       cmap='RdYlBu_r',
                       xticklabels=[v.replace('Log_', '').replace('Lab_', '') for v in variables],
                       yticklabels=False,
                       figsize=(15, 10),
                       ax=ax2)
    
    plt.suptitle('Hierarchical Clustering of Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}hierarchical_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()