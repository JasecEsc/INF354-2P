CREATE OR REPLACE FUNCTION public.diferencia_entre_cadenas(
    cadena1 character varying,
    cadena2 character varying
)
RETURNS integer
LANGUAGE plpgsql
AS $function$
DECLARE
    longitud_diferencia integer := 0;
    longitud_minima integer := LEAST(LENGTH(cadena1), LENGTH(cadena2));
    i integer := 1;
BEGIN
    WHILE i <= longitud_minima LOOP
        IF SUBSTRING(cadena1 FROM i FOR 1) <> SUBSTRING(cadena2 FROM i FOR 1) THEN
            longitud_diferencia := longitud_diferencia + 1;
        END IF;

        i := i + 1;
    END LOOP;

    -- Agrega la longitud de la diferencia entre las cadenas de longitud diferente
    longitud_diferencia := longitud_diferencia + ABS(LENGTH(cadena1) - LENGTH(cadena2));

    RETURN longitud_diferencia;
END;
$function$;

-- Ejemplo de uso
SELECT public.diferencia_entre_cadenas('orismendi', 'arizmendi') AS longitud_diferencia;

