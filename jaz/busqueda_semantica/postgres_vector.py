import psycopg2
import numpy as np
from typing import List, Tuple, Any, Optional, Union

class PostgresVector:
    def __init__(self, host="localhost", port=5434, database="ferreteria", user="postgres", password="postgres"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cur = None
        
    def open_db(self):
        """Abre una conexión a la base de datos"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            return True
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            return False
        
    def open_cursor(self):
        """Abre un cursor para la conexión actual"""
        if self.conn is None:
            return False
        try:
            self.cur = self.conn.cursor()
            return True
        except Exception as e:
            print(f"Error al abrir el cursor: {e}")
            return False

    def close_cursor(self):
        """Cierra el cursor actual si está abierto"""
        if self.cur is not None:
            try:
                self.cur.close()
                self.cur = None
                return True
            except Exception as e:
                print(f"Error al cerrar el cursor: {e}")
                return False
        return True

    def close_db(self):
        """Cierra la conexión a la base de datos si está abierta"""
        if self.conn is not None:
            try:
                self.conn.close()
                self.conn = None
                return True
            except Exception as e:
                print(f"Error al cerrar la conexión: {e}")
                return False
        return True

    def execute_query(self, sql: str, params: Optional[tuple] = None, fetch: bool = False) -> Union[List[Tuple], bool]:
        """
        Ejecuta una consulta SQL con manejo de transacciones
        
        Args:
            sql: Consulta SQL a ejecutar
            params: Parámetros para la consulta (opcional)
            fetch: Si es True, devuelve los resultados de la consulta
            
        Returns:
            Lista de resultados si fetch=True, o booleano indicando éxito
        """
        if self.conn is None or self.cur is None:
            print("No hay conexión o cursor abierto")
            return False if not fetch else []
            
        try:
            # Imprimir información de depuración
            # print(f"SQL: {sql}")
            '''
            if params:
                print(f"Params: {params}")
                print(f"Tipos de params: {[type(p).__name__ for p in params]}")
            '''
                
            if params:
                # Verificar si hay algún ellipsis en los parámetros
                if any(p is ... for p in params):
                    print("¡ADVERTENCIA! Se encontró un ellipsis en los parámetros")
                    # Reemplazar ellipsis con None
                    params = tuple(None if p is ... else p for p in params)
                    print(f"Params corregidos: {params}")
                
                self.cur.execute(sql, params)
            else:
                self.cur.execute(sql)
                
            if fetch:
                result = self.cur.fetchall()
                self.conn.commit()
                return result
            else:
                self.conn.commit()
                return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error en la consulta: {e}")
            print(f"SQL: {sql}")
            if params:
                print(f"Params: {params}")
                print(f"Tipos de params: {[type(p).__name__ for p in params]}")
            return False if not fetch else []
        
    # --- MÉTODOS CRUD BÁSICOS ---
    
    def create_vector_extension(self) -> bool:
        """Crea la extensión vector si no existe"""
        return self.execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
        
    def create_items_table(self, dimension: int = 384) -> bool:
        """
        Crea la tabla items con soporte para vectores de la dimensión especificada
        
        Args:
            dimension: Dimensión de los vectores a almacenar
        """
        sql = f"""
        CREATE TABLE IF NOT EXISTS items (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            embedding vector({dimension}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        return self.execute_query(sql)
    
    # --- CREATE ---
    
    def insert_item(self, name: str, description: str, embedding: Union[List[float], np.ndarray]) -> bool:
        """
        Inserta un nuevo item con su vector de embedding
        
        Args:
            name: Nombre del item
            description: Descripción del item
            embedding: Vector de embedding (lista o numpy array)
            
        Returns:
            True si la inserción fue exitosa, False en caso contrario
        """
        # Convertir a lista si es un array numpy
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        sql = """
        INSERT INTO items (name, description, embedding) 
        VALUES (%s, %s, %s::vector)
        """
        return self.execute_query(sql, (name, description, embedding))
    
    def insert_many_items(self, items: List[Tuple[str, str, Union[List[float], np.ndarray]]]) -> bool:
        """
        Inserta múltiples items en una sola transacción
        
        Args:
            items: Lista de tuplas (name, description, embedding)
            
        Returns:
            True si todas las inserciones fueron exitosas, False en caso contrario
        """
        if not items:
            return True
            
        try:
            # Iniciar transacción explícita
            if self.conn is None or self.cur is None:
                return False
                
            for name, description, embedding in items:
                # Convertir a lista si es un array numpy
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                sql = """
                INSERT INTO items (name, description, embedding) 
                VALUES (%s, %s, %s::vector)
                """
                self.cur.execute(sql, (name, description, embedding))
                
            # Confirmar transacción
            self.conn.commit()
            return True
        except Exception as e:
            # Revertir transacción en caso de error
            if self.conn:
                self.conn.rollback()
            print(f"Error al insertar múltiples items: {e}")
            return False
    
    # --- READ ---
    
    def get_item_by_id(self, item_id: int) -> Optional[Tuple]:
        """
        Obtiene un item por su ID
        
        Args:
            item_id: ID del item a buscar
            
        Returns:
            Tupla con los datos del item o None si no se encuentra
        """
        sql = "SELECT id, name, description, embedding FROM items WHERE id = %s"
        result = self.execute_query(sql, (item_id,), fetch=True)
        return result[0] if result else None
    
    def get_all_items(self, limit: int = 100, offset: int = 0) -> List[Tuple]:
        """
        Obtiene todos los items con paginación
        
        Args:
            limit: Número máximo de items a devolver
            offset: Número de items a saltar
            
        Returns:
            Lista de tuplas con los datos de los items
        """
        sql = """
        SELECT id, name, description FROM items
        ORDER BY id
        LIMIT %s OFFSET %s
        """
        return self.execute_query(sql, (limit, offset), fetch=True)
    
    def search_by_text(self, text: str, limit: int = 10) -> List[Tuple]:
        """
        Busca items por texto en nombre o descripción
        
        Args:
            text: Texto a buscar
            limit: Número máximo de resultados
            
        Returns:
            Lista de tuplas con los items encontrados
        """
        sql = """
        SELECT id, name, description 
        FROM items 
        WHERE name ILIKE %s OR description ILIKE %s
        LIMIT %s
        """
        search_pattern = f"%{text}%"
        return self.execute_query(sql, (search_pattern, search_pattern, limit), fetch=True)
    
    def search_similar(self, query_vector: Union[List[float], np.ndarray], limit: int = 5) -> List[Tuple]:
        """
        Busca items similares al vector de consulta usando distancia coseno
        
        Args:
            query_vector: Vector de consulta (lista o numpy array)
            limit: Número máximo de resultados
            
        Returns:
            Lista de tuplas (id, name, description, distancia)
        """
        # Convertir a lista si es un array numpy
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
            
        sql = """
        SELECT id, name, description, embedding <=> %s::vector AS distance
        FROM items
        ORDER BY distance
        LIMIT %s
        """
        return self.execute_query(sql, (query_vector, limit), fetch=True)
    
    # --- UPDATE ---
    
    def update_item(self, item_id: int, name: Optional[str] = None, 
                   description: Optional[str] = None, 
                   embedding: Optional[Union[List[float], np.ndarray]] = None) -> bool:
        """
        Actualiza un item existente
        
        Args:
            item_id: ID del item a actualizar
            name: Nuevo nombre (opcional)
            description: Nueva descripción (opcional)
            embedding: Nuevo vector de embedding (opcional)
            
        Returns:
            True si la actualización fue exitosa, False en caso contrario
        """
        # Verificar que al menos un campo se va a actualizar
        if name is None and description is None and embedding is None:
            return False
            
        try:
            # Iniciar transacción explícita
            if self.conn is None or self.cur is None:
                return False
                
            # Construir la consulta dinámicamente según los campos a actualizar
            update_parts = []
            params = []
            
            if name is not None:
                update_parts.append("name = %s")
                params.append(name)
                
            if description is not None:
                update_parts.append("description = %s")
                params.append(description)
                
            if embedding is not None:
                # Convertir a lista si es un array numpy
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                update_parts.append("embedding = %s::vector")
                params.append(embedding)
                
            # Añadir timestamp de actualización
            update_parts.append("updated_at = CURRENT_TIMESTAMP")
            
            # Construir la consulta final
            sql = f"""
            UPDATE items 
            SET {', '.join(update_parts)}
            WHERE id = %s
            """
            params.append(item_id)
            
            self.cur.execute(sql, tuple(params))
            
            # Verificar si se actualizó alguna fila
            rows_affected = self.cur.rowcount
            
            # Confirmar transacción
            self.conn.commit()
            
            return rows_affected > 0
        except Exception as e:
            # Revertir transacción en caso de error
            if self.conn:
                self.conn.rollback()
            print(f"Error al actualizar item: {e}")
            return False
    
    # --- DELETE ---
    
    def delete_item(self, item_id: int) -> bool:
        """
        Elimina un item por su ID
        
        Args:
            item_id: ID del item a eliminar
            
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        sql = "DELETE FROM items WHERE id = %s"
        try:
            if self.conn is None or self.cur is None:
                return False
                
            self.cur.execute(sql, (item_id,))
            rows_affected = self.cur.rowcount
            self.conn.commit()
            return rows_affected > 0
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error al eliminar item: {e}")
            return False
    
    def delete_all_items(self) -> bool:
        """
        Elimina todos los items de la tabla (¡CUIDADO!)Params:
        
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        sql = "DELETE FROM items"
        try:
            if self.conn is None or self.cur is None:
                return False
                
            self.cur.execute(sql)
            self.conn.commit()
            return True
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error al eliminar todos los items: {e}")
            return False
    
    # --- MÉTODOS DE CONVENIENCIA ---
    
    def connect(self):
        """Método de conveniencia para abrir conexión y cursor"""
        success = self.open_db()
        if success:
            success = self.open_cursor()
        return self if success else None
        
    def disconnect(self):
        """Método de conveniencia para cerrar cursor y conexión"""
        self.close_cursor()
        self.close_db()
    
    # Alias para mantener compatibilidad con código existente
    def insert_vector(self, name: str, description: str, embedding: Union[List[float], np.ndarray]) -> bool:
        """Alias de insert_item para mantener compatibilidad con código existente"""
        return self.insert_item(name, description, embedding)
        
    def __enter__(self):
        """Soporte para el uso con 'with'"""
        return self.connect()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Soporte para el uso con 'with'"""
        if exc_type is not None:
            # Si hubo una excepción, hacer rollback
            if self.conn:
                self.conn.rollback()
        self.disconnect()
