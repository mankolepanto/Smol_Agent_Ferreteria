FROM ghcr.io/postgresml/postgresml:2.9.3
  
RUN apt-get update && \
    apt-get install -y postgresql-plpython3-15 postgresql-15-pgvector && \
    rm -rf /var/lib/apt/lists/*

# Install InstructorEmbedding as the postgres user
USER postgres
RUN pip install InstructorEmbedding

# Crear script de inicialización para la base de datos
RUN echo 'CREATE DATABASE ferreteria WITH OWNER postgres;' > /tmp/init-db.sql

# Switch back to root for the restart command
USER root

# Asegurar que el directorio de datos exista y tenga los permisos correctos
RUN mkdir -p /var/lib/postgresql/data/pgdata && \
    chown -R postgres:postgres /var/lib/postgresql/data && \
    chmod 700 /var/lib/postgresql/data

# Crear el script de inicio en un archivo separado
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'if [ -z "$(ls -A /var/lib/postgresql/data/pgdata)" ]; then' >> /entrypoint.sh && \
    echo '  echo "Inicializando directorio de datos de PostgreSQL..."' >> /entrypoint.sh && \
    echo '  su postgres -c "initdb -D /var/lib/postgresql/data/pgdata"' >> /entrypoint.sh && \
    echo '  echo "data_directory = '"'"'/var/lib/postgresql/data/pgdata'"'"'" >> /var/lib/postgresql/data/pgdata/postgresql.conf' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'service postgresql start' >> /entrypoint.sh && \
    echo 'sleep 5' >> /entrypoint.sh && \
    echo '# Verificar si la base de datos ferreteria existe' >> /entrypoint.sh && \
    echo 'if ! su postgres -c "psql -l" | grep -q ferreteria; then' >> /entrypoint.sh && \
    echo '  echo "Creando base de datos ferreteria..."' >> /entrypoint.sh && \
    echo '  su postgres -c "psql -f /tmp/init-db.sql"' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '  echo "La base de datos ferreteria ya existe"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo 'echo "PostgreSQL iniciado y base de datos ferreteria verificada"' >> /entrypoint.sh && \
    echo 'tail -f /dev/null' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Establecer el punto de entrada
ENTRYPOINT ["/entrypoint.sh"]