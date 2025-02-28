FROM ghcr.io/postgresml/postgresml:2.9.3
  
RUN apt-get update && \
    apt-get install -y postgresql-plpython3-15 postgresql-15-pgvector && \
    rm -rf /var/lib/apt/lists/*

# Install InstructorEmbedding as the postgres user
USER postgres
RUN pip install InstructorEmbedding

# Crear script de inicialización para la base de datos
COPY <<EOF /tmp/init-db.sql
CREATE DATABASE ferreteria WITH OWNER postgres;
EOF

# Switch back to root for the restart command
USER root

# Crear un script de inicio
RUN echo '#!/bin/bash\n\
service postgresql start\n\
sleep 5\n\
su postgres -c "psql -f /tmp/init-db.sql" || true\n\
echo "PostgreSQL iniciado y base de datos ferreteria creada (si no existía)"\n\
tail -f /dev/null' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Establecer el punto de entrada
ENTRYPOINT ["/entrypoint.sh"]
