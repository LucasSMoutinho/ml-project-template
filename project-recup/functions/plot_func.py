import matplotlib.pyplot as plt
import pandas as pd

def plot_vol(df, seg, col_dat='ano_mes_dia_rfrc'):
    # 1. Filtre o DataFrame para o segmento desejado
    segmento_df = df[df['desc_dom_neg'] == seg].copy()

    # 2. CONVERTE PARA DATETIME (Garante a ordenação cronológica correta)
    segmento_df[col_dat] = pd.to_datetime(segmento_df[col_dat], format='%d/%m/%Y')

    # 3. Agrupe por data e conte a volumetria
    volumetria_por_data = (
        segmento_df.groupby(col_dat)
        .size() 
        .reset_index(name='volumetria') 
        .sort_values(col_dat) # Agora sim, ordena cronologicamente de verdade
    )

    # 4. Plote o gráfico de linha
    plt.figure(figsize=(12, 6))
    plt.plot(
        volumetria_por_data[col_dat], 
        volumetria_por_data['volumetria'], 
        marker='o',          
        linestyle='-',       
        color='b'            
    )

    # Estilização do gráfico
    plt.title(f'Volumetria ao longo do tempo - Segmento {seg}', fontsize=14)
    plt.xlabel(f'Data ({col_dat})', fontsize=12)
    plt.ylabel('Quantidade de Registros', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=90) 
    plt.tight_layout()

    plt.show()
    

def plot_vol_target(df, seg, col_dat='ano_mes_dia_rfrc'):
    # 1. Filtre o DataFrame para o segmento desejado
    segmento_df = df[df['desc_dom_neg'] == seg].copy()

    # 2. Converte a coluna de data para datetime e garante a ordenação
    segmento_df[col_dat] = pd.to_datetime(segmento_df[col_dat], format='%d/%m/%Y')

    # 3. Agrupe por data e target, contando os registros
    volumetria_target = (
        segmento_df.groupby([col_dat, 'target'])
        .size()
        .unstack(fill_value=0) # Transforma os valores da target (0 e 1) em colunas separadas
        .sort_index()          # Garante a ordenação cronológica das datas
    )

    # 4. Plote o gráfico com duas linhas (uma para cada target)
    plt.figure(figsize=(12, 6))
    
    # Linha para a Target 0
    if 0 in volumetria_target.columns:
        plt.plot(
            volumetria_target.index, 
            volumetria_target[0], 
            marker='o', 
            linestyle='-', 
            label='Target 0',
            color='tab:blue'
        )
        
    # Linha para a Target 1
    if 1 in volumetria_target.columns:
        plt.plot(
            volumetria_target.index, 
            volumetria_target[1], 
            marker='o', 
            linestyle='-', 
            label='Target 1',
            color='tab:orange'
        )

    # Estilização do gráfico
    plt.title(f'Estabilidade da Target ao longo do tempo - Segmento {seg}', fontsize=14)
    plt.xlabel(f'Data ({col_dat})', fontsize=12)
    plt.ylabel('Volumetria (Quantidade)', fontsize=12)
    plt.legend(title='Classes da Target')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.show()